import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel
import numpy as np
import os

import config

class ImageEncoder(nn.Module):
    """
    Encodes images using a pretrained model.
    Prioritizes loading a local resnet-50 model if available.
    """
    def __init__(self, model_name=config.IMAGE_ENCODER, pretrained=True):
        super().__init__()
        
        local_img_model_dir = "/devdata/models/resnet-50/"
        
        self.is_hf_model = False
        # Check if the requested model is 'resnet50' and a local copy exists.
        if model_name == 'resnet50' and os.path.isdir(local_img_model_dir):
            print(f"Attempting to load local resnet-50 model from Hugging Face directory: {local_img_model_dir}")
            # This is a Hugging Face model, load it with AutoModel, which reads config.json
            self.model = AutoModel.from_pretrained(local_img_model_dir)
            # The feature dimension is in the config file. For ResNet, it's the last hidden size.
            self.feature_dim = self.model.config.hidden_sizes[-1]
            self.is_hf_model = True
            print("Successfully loaded local model using Hugging Face's AutoModel.")
        else:
            # Fallback to default timm online behavior
            if not model_name:
                raise ValueError("config.IMAGE_ENCODER is empty, and no local resnet50 model was found.")
            print(f"Loading '{model_name}' from timm's online repository.")
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.model.num_features

    def forward(self, x):
        if self.is_hf_model:
            # Hugging Face vision models return BaseModelOutputWithPooling
            # last_hidden_state is the feature map from the last stage
            outputs = self.model(x)
            feature_map = outputs.last_hidden_state
            # Perform global average pooling to get a feature vector.
            return F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze(-1).squeeze(-1)
        else:
            # Default timm forward pass (already returns a feature vector)
            return self.model(x)

class TextEncoder(nn.Module):
    """
    Encodes text using a pretrained model from Hugging Face Transformers.
    """
    def __init__(self, model_name=config.TEXT_ENCODER):
        super().__init__()
        local_txt_model_dir = "/devdata/models/distilbert-base-uncased/"
        self.model = AutoModel.from_pretrained(local_txt_model_dir)
        self.feature_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token's representation
        return outputs.last_hidden_state[:, 0, :]

class ModalityExpertAttention(nn.Module):
    """
    Modality-specific Expert Attention Heads for MEE Module.
    Each attention head captures modality-specific semantics while preserving structural distinctiveness.
    """
    def __init__(self, embed_dim, num_heads, num_modalities):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        
        # Modality-specific attention heads
        self.modality_heads = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_modalities)
        ])
        
        # Modality-specific projection layers
        self.modality_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(num_modalities)
        ])
        
        # Router for dynamic modality weighting
        self.router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, img_features, text_features, modalities):
        batch_size = img_features.size(0)
        
        # Compute modality weights using router
        combined_features = img_features + text_features
        modality_weights = self.router(combined_features)  # [B, num_modalities]
        
        # Apply modality-specific attention
        modality_outputs = []
        for i in range(self.num_modalities):
            # Modality-specific cross-attention
            head_output, _ = self.modality_heads[i](
                img_features.unsqueeze(1),  # Query
                text_features.unsqueeze(1),  # Key
                text_features.unsqueeze(1)   # Value
            )
            head_output = head_output.squeeze(1)
            
            # Modality-specific projection
            head_output = self.modality_projections[i](head_output)
            modality_outputs.append(head_output)
        
        # Weighted combination of modality-specific outputs
        modality_outputs = torch.stack(modality_outputs, dim=1)  # [B, num_modalities, embed_dim]
        weighted_output = torch.sum(modality_outputs * modality_weights.unsqueeze(-1), dim=1)
        
        return weighted_output, modality_weights

class PrototypeBank(nn.Module):
    """
    Prototype Bank for PDE Module - maintains class prototypes for contrastive learning.
    """
    def __init__(self, num_classes, feature_dim, momentum=0.99):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        # Initialize prototypes randomly
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
        
        # Moving average for prototype updates
        self.register_buffer('prototype_momentum', torch.zeros(num_classes, feature_dim))
        self.register_buffer('prototype_count', torch.zeros(num_classes))

    def update_prototypes(self, features, labels):
        """Update prototypes using exponential moving average."""
        for i in range(self.num_classes):
            class_mask = (labels == i)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                class_mean = class_features.mean(dim=0)
                
                # Update moving average
                self.prototype_momentum[i] = (
                    self.momentum * self.prototype_momentum[i] + 
                    (1 - self.momentum) * class_mean
                )
                self.prototype_count[i] += class_mask.sum()
                
                # Update prototype
                if self.prototype_count[i] > 0:
                    self.prototypes.data[i] = F.normalize(
                        self.prototype_momentum[i] / self.prototype_count[i], dim=-1
                    )

    def get_prototypes(self):
        """Get current prototypes."""
        return self.prototypes

class ProjectionHead(nn.Module):
    """A simple projection head for contrastive learning."""
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.head(x), dim=-1)

class MainModel(nn.Module):
    """
    FICMD_Med: A Novel Framework for Imbalanced Class and Modality Distributions of Medical Images.
    
    The framework integrates:
    1. PDE Module: Prototype-guided Discriminative Enhancement
    2. MEE Module: Modality-focus Expert Enhancement  
    3. ASGO Module: Adaptive Sample-level Gradient Optimization (handled in optimizer.py)
    """
    def __init__(self, num_classes, num_modalities, modality_counts, img_modalities):
        super().__init__()
        self.img_modalities = img_modalities  # List of image modality names, e.g., ['MRI', 'CT']
        self.num_img_modalities = len(img_modalities)
        self.num_classes = num_classes
        self.num_modalities = num_modalities
        
        # === PDE Module Components ===
        # Encoders for contrastive learning branch
        self.image_encoder_pde = ImageEncoder()
        self.text_encoder_pde = TextEncoder()
        
        # Prototype bank for class prototypes
        self.prototype_bank = PrototypeBank(
            num_classes=num_classes,
            feature_dim=config.EMBEDDING_DIM
        )
        
        # Projection heads for PDE branch
        self.img_proj_pde = nn.ModuleList([
            ProjectionHead(self.image_encoder_pde.feature_dim, config.EMBEDDING_DIM)
            for _ in range(self.num_img_modalities)
        ])
        self.text_proj_pde = ProjectionHead(self.text_encoder_pde.feature_dim, config.EMBEDDING_DIM)
        
        # === MEE Module Components ===
        # Encoders for modality expert branch
        self.image_encoder_mee = ImageEncoder()
        self.text_encoder_mee = TextEncoder()
        
        # Modality expert attention
        self.modality_expert_attention = ModalityExpertAttention(
            embed_dim=config.EMBEDDING_DIM,
            num_heads=config.NUM_CROSS_ATTENTION_HEADS,
            num_modalities=num_modalities
        )
        
        # Projection heads for MEE branch
        self.img_proj_mee = nn.ModuleList([
            ProjectionHead(self.image_encoder_mee.feature_dim, config.EMBEDDING_DIM)
            for _ in range(self.num_img_modalities)
        ])
        self.text_proj_mee = ProjectionHead(self.text_encoder_mee.feature_dim, config.EMBEDDING_DIM)
        
        # === Classification Heads ===
        # PDE branch classifier
        self.classifier_pde = nn.Linear(config.EMBEDDING_DIM, num_classes)
        
        # MEE branch classifier  
        self.classifier_mee = nn.Linear(config.EMBEDDING_DIM, num_classes)
        
        # === Modality Weighting ===
        if config.USE_MODALITY_WEIGHTS:
            total_samples = sum(modality_counts.values())
            weights = torch.tensor([total_samples / modality_counts.get(i, 1) for i in range(num_modalities)])
            self.modality_weights = F.softmax(weights, dim=0) * num_modalities
        else:
            self.modality_weights = torch.ones(num_modalities)

    def forward(self, batch):
        images = batch['image']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        modalities = batch['modality']
        img_modality = batch['img_modality']  # shape: (B,)
        labels = batch['label']
        
        # === PDE Module Forward Pass ===
        # Image and text encoding
        img_embed_pde = self.image_encoder_pde(images)
        text_embed_pde = self.text_encoder_pde(input_ids, attention_mask)
        
        # Modality-specific projections
        img_proj_pde = torch.stack([
            self.img_proj_pde[m](img_embed_pde[i].unsqueeze(0)).squeeze(0)
            for i, m in enumerate(img_modality)
        ])
        text_proj_pde = self.text_proj_pde(text_embed_pde)
        
        # Update prototypes
        self.prototype_bank.update_prototypes(img_proj_pde, labels)
        
        # PDE branch prediction
        y_predict_pde = self.classifier_pde(img_proj_pde)
        
        # === MEE Module Forward Pass ===
        # Image and text encoding
        img_embed_mee = self.image_encoder_mee(images)
        text_embed_mee = self.text_encoder_mee(input_ids, attention_mask)
        
        # Modality-specific projections
        img_proj_mee = torch.stack([
            self.img_proj_mee[m](img_embed_mee[i].unsqueeze(0)).squeeze(0)
            for i, m in enumerate(img_modality)
        ])
        text_proj_mee = self.text_proj_mee(text_embed_mee)
        
        # Modality expert attention
        mee_output, modality_weights = self.modality_expert_attention(
            img_proj_mee, text_proj_mee, modalities
        )
        
        # MEE branch prediction
        y_predict_mee = self.classifier_mee(mee_output)
        
        return {
            'y_predict1': y_predict_pde,  # PDE branch output
            'y_predict2': y_predict_mee,  # MEE branch output
            'image_embedding': img_proj_pde,  # For IIC loss
            'image_projection': img_proj_pde,  # For ITC loss
            'text_projection': text_proj_pde,  # For ITC loss
            'prototypes': self.prototype_bank.get_prototypes(),  # For prototype-based loss
            'modality_weights': modality_weights,  # For modality weighting
        }

    def get_logits(self, images, input_ids, attention_mask, modalities, img_modality):
        """Get logits from the main PDE branch for inference."""
        img_embed_pde = self.image_encoder_pde(images)
        text_embed_pde = self.text_encoder_pde(input_ids, attention_mask)
        
        img_proj_pde = torch.stack([
            self.img_proj_pde[m](img_embed_pde[i].unsqueeze(0)).squeeze(0)
            for i, m in enumerate(img_modality)
        ])
        
        y_predict_pde = self.classifier_pde(img_proj_pde)
        return y_predict_pde
