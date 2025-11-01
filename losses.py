import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class ITCLoss(nn.Module):
    """Image-Text Contrastive Loss"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        # image_features: [batch_size, embed_dim]
        # text_features: [batch_size, embed_dim]
        
        # Calculate logits
        logits = (image_features @ text_features.T) / self.temperature
        
        # Create labels
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Calculate loss
        loss_img = self.criterion(logits, labels)
        loss_txt = self.criterion(logits.T, labels)
        
        return (loss_img + loss_txt) / 2

class IICLoss(nn.Module):
    """
    Image-Image Contrastive Loss (Supervised) with Prototype and P(xi) weighting.
    Implements the PDE Module's prototype-guided contrastive learning as described in the paper.
    """
    def __init__(self, temperature=0.07, num_classes=3, feature_dim=512):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Prototype mechanism for minority class enhancement
        if config.USE_IIC_PROTOTYPES:
            self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
            self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
    
    def update_prototypes(self, features, labels):
        """Update prototypes using exponential moving average."""
        if not config.USE_IIC_PROTOTYPES:
            return
            
        alpha = config.PROTOTYPE_UPDATE_ALPHA
        for i in range(self.num_classes):
            class_features = features[labels == i]
            if len(class_features) > 0:
                # Update prototype with moving average
                new_prototype = class_features.mean(dim=0)
                self.prototypes.data[i] = alpha * self.prototypes.data[i] + (1 - alpha) * new_prototype
                # Normalize prototype
                self.prototypes.data[i] = F.normalize(self.prototypes.data[i], dim=-1)

    def forward(self, features, labels, p_xi=None, prototypes=None):
        """
        Forward pass implementing prototype-guided contrastive learning.
        
        Args:
            features: [batch_size, embed_dim] - image embeddings
            labels: [batch_size] - class labels
            p_xi: [batch_size] - sample weights for imbalance-aware strategy
            prototypes: [num_classes, embed_dim] - class prototypes (optional)
        """
        batch_size = features.shape[0]
        features = F.normalize(features, dim=-1)
        
        # Create positive/negative masks based on class labels
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        # Handle cases with no positive samples
        positive_counts = mask.sum(1)
        no_positive_mask = (positive_counts == 0)
        
        if no_positive_mask.any():
            self_mask = torch.eye(batch_size, device=features.device)
            mask = mask + no_positive_mask.unsqueeze(1).float() * self_mask
            logits_mask = logits_mask + no_positive_mask.unsqueeze(1).float() * self_mask

        # Prototype-guided enhancement
        if config.USE_IIC_PROTOTYPES and prototypes is not None:
            # Normalize prototypes
            proto_features = F.normalize(prototypes, dim=-1)
            
            # Similarity between batch features and prototypes
            feat_proto_sim = torch.div(
                torch.matmul(features, proto_features.T),
                self.temperature
            ) # (B, num_classes)
            
            # Create prototype masks: same-class prototype is positive, others are negative
            proto_pos_mask = F.one_hot(labels.squeeze(-1), num_classes=self.num_classes).float()
            
            # Augment logits and masks with prototypes
            logits = torch.cat([logits, feat_proto_sim], dim=1)
            mask = torch.cat([mask, proto_pos_mask], dim=1)
            augmented_neg_mask = torch.ones(batch_size, self.num_classes).to(features.device)
            logits_mask = torch.cat([logits_mask, augmented_neg_mask], dim=1)

        # Compute contrastive loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        epsilon = 1e-8
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + epsilon)
        
        # Calculate loss
        loss = -mean_log_prob_pos
        
        # Apply P(xi) weighting for imbalance-aware strategy
        if p_xi is not None:
            loss = p_xi * loss
        
        return loss.mean()

class ConsistencyLoss(nn.Module):
    """
    Consistency regularizer using KL Divergence.
    """
    def __init__(self):
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, y_predict1, y_predict2):
        prob1 = F.log_softmax(y_predict1, dim=1)
        prob2 = F.softmax(y_predict2, dim=1)
        return self.kl_div(prob1, prob2.detach()) # Detach one side to prevent collapse

def get_p_xi(labels, modalities, class_counts, modality_counts):
    """
    Calculates the P(xi) weighting factor based on class and modality rarity.
    Implements the imbalance-aware sample weighting strategy from the PDE Module.
    
    Args:
        labels: [batch_size] - class labels
        modalities: [batch_size] - modality labels  
        class_counts: dict - count of samples per class
        modality_counts: dict - count of samples per modality
        
    Returns:
        p_xi: [batch_size] - sample weights for imbalance-aware strategy
    """
    if config.P_XI_FACTOR == 0:
        return torch.zeros(len(labels))
    if config.P_XI_FACTOR == 1:
        return torch.ones(len(labels))
    
    # Calculate class and modality rarity factors
    total_class_samples = sum(class_counts.values())
    total_modality_samples = sum(modality_counts.values())
    
    # Class rarity: inverse frequency weighting
    class_rarity = torch.tensor([
        total_class_samples / class_counts.get(l.item(), 1) 
        for l in labels
    ])
    
    # Modality rarity: inverse frequency weighting  
    modality_rarity = torch.tensor([
        total_modality_samples / modality_counts.get(m.item(), 1) 
        for m in modalities
    ])
    
    # Combine class and modality rarity as described in the paper
    p_xi = class_rarity * modality_rarity
    
    # Normalize to [0, 1] range for stable training
    p_xi = (p_xi - p_xi.min()) / (p_xi.max() - p_xi.min() + 1e-8)
    
    return p_xi.to(config.DEVICE)