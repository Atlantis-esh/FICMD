import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer

import config

class MedicalImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading medical images, captions, and metadata.
    Handles tokenization of text and image transformations.
    """
    def __init__(self, annotations_df, img_dir, tokenizer, transform, modality_map, class_map):
        self.annotations_df = annotations_df
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.modality_map = modality_map
        self.class_map = class_map

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.annotations_df.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_name}. Skipping.")
            # Return a dummy sample or handle it as you see fit
            # For simplicity, we might skip this sample in the collate_fn or return None
            # A better approach is to clean the dataframe before creating the dataset
            return None


        # print("Original image size:", image.size) 
        # Image transformations
        if self.transform:
            image = self.transform(image)
        # print("Transformed size:", image.shape) 

        
        # Text data and tokenization
        caption = self.annotations_df.iloc[idx, 3]
        encoded_caption = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=128, # Max length for BERT-like models
            return_tensors='pt'
        )
        input_ids = encoded_caption['input_ids'].squeeze(0)
        attention_mask = encoded_caption['attention_mask'].squeeze(0)

        # Labels and metadata
        label_str = self.annotations_df.iloc[idx, 2]
        label = self.class_map[label_str]
        modality_str = self.annotations_df.iloc[idx, 1]
        modality = self.modality_map[modality_str]
        img_modality = modality  # Assume modality_map is the image modality index

        sample = {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'modality': torch.tensor(modality, dtype=torch.long),
            'img_modality': torch.tensor(img_modality, dtype=torch.long),
            'metadata': {
                'image_path': img_name,
                'caption': caption
            }
        }
        return sample

def get_transforms(img_size):
    """Returns a composition of image transformations for training and validation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_dataloaders():
    """
    Creates and returns train, validation, and test dataloaders.
    """
    # 1. Load annotations and prepare mappings
    annotations_df = pd.read_csv(config.CAPTIONS_FILE)
    

    # Filter out missing files
    original_len = len(annotations_df)
    annotations_df['exists'] = annotations_df['image'].apply(lambda x: os.path.exists(os.path.join(config.DATA_DIR, 'images', x)))
    annotations_df = annotations_df[annotations_df['exists']].drop(columns=['exists'])
    if len(annotations_df) < original_len:
        print(f"Warning: Dropped {original_len - len(annotations_df)} samples due to missing image files.")

    # Create modality and class mappings
    modalities = sorted(annotations_df['modality'].unique())
    classes = sorted(annotations_df['label'].unique())
    modality_map = {modality: i for i, modality in enumerate(modalities)}
    class_map = {label: i for i, label in enumerate(classes)}
    
    # 2. Initialize tokenizer and transforms
    tokenizer = AutoTokenizer.from_pretrained("/devdata/models/distilbert-base-uncased/")
    transform = get_transforms(config.IMG_SIZE)

    # 3. Create full dataset
    full_dataset = MedicalImageDataset(
        annotations_df=annotations_df,
        img_dir=os.path.join(config.DATA_DIR, 'images'),
        tokenizer=tokenizer,
        transform=transform,
        modality_map=modality_map,
        class_map=class_map
    )

    # 4. Split dataset (80% train, 20% test)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.seed(42) # for reproducibility
    np.random.shuffle(indices)

    train_end = int(np.floor(0.8 * dataset_size))
    
    train_indices = indices[:train_end]
    test_indices = indices[train_end:]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # 5. Create dataloaders
    def collate_fn(batch):
        # Filter out None samples
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)
    print(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn
    )
    
    print(f"Data loaded. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    metadata = {
        'modality_map': modality_map,
        'class_map': class_map,
        'num_classes': len(classes),
        'num_modalities': len(modalities),
        'train_df': annotations_df.iloc[train_indices]
    }

    return train_loader, test_loader, metadata

if __name__ == '__main__':
    # For testing purposes
    train_loader, test_loader, metadata = get_dataloaders()
    
    print("\n--- Metadata ---")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Class map: {metadata['class_map']}")
    print(f"Number of modalities: {metadata['num_modalities']}")
    print(f"Modality map: {metadata['modality_map']}")

    print("\n--- Train Loader Sample Batch ---")
    sample_batch = next(iter(train_loader))
    if sample_batch:
        print("Image shape:", sample_batch['image'].shape)
        print("Input IDs shape:", sample_batch['input_ids'].shape)
        print("Attention Mask shape:", sample_batch['attention_mask'].shape)
        print("Label shape:", sample_batch['label'].shape)
        print("Modality shape:", sample_batch['modality'].shape)
    else:
        print("Could not retrieve a batch from the train loader.") 