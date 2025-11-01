import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, f1_score
import torch.nn.functional as F
# Local imports
import config
from dataset import get_dataloaders
from model import MainModel
from losses import ITCLoss, IICLoss, ConsistencyLoss, get_p_xi
from optimizer import balanced_gradient_update, gradient_balanced_grouping, combined_gradient_update
import random
import os


def set_seed(seed=42):
    """Set random seed to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)


class Trainer:
    def __init__(self):
        # --- Set random seed ---
        set_seed(config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else 42)
        # --- Setup ---
        self.device = config.DEVICE
        print(f"Using device: {self.device}")
        print(f"Random seed set to: {config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else 42}")
        print(f"Dynamic gradient optimization: {'Enabled' if config.USE_DYNAMIC_GRADIENT else 'Disabled'}")
        
        # --- Dynamic gradient optimization parameters ---
        if config.USE_DYNAMIC_GRADIENT:
            self.regroup_interval = config.REGROUP_INTERVAL
            self.n_groups = config.NUM_GRADIENT_GROUPS
            self.normalization_type = config.GRADIENT_NORMALIZATION_TYPE
            self.class_weight = 1.0 - config.W_MIX
            self.sample_weight = config.W_MIX

        # --- Data ---
        self.train_loader, self.test_loader, self.metadata = get_dataloaders()
        self.class_counts = Counter(self.metadata['train_df']['label'])
        self.modality_counts = Counter(self.metadata['train_df']['modality'].map(self.metadata['modality_map']))
        self.img_modalities = list(self.metadata['modality_map'].keys())

        # --- Model ---
        self.model = MainModel(
            num_classes=self.metadata['num_classes'],
            num_modalities=self.metadata['num_modalities'],
            modality_counts={k: v for k,v in self.modality_counts.items()},
            img_modalities=self.img_modalities
        ).to(self.device)

        # --- Optimizer & Scheduler ---
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5, factor=0.5)

        # --- Loss Functions ---
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_itc = ITCLoss().to(self.device)
        self.criterion_iic = IICLoss(
            num_classes=self.metadata['num_classes'], 
            feature_dim=config.EMBEDDING_DIM
        ).to(self.device)
        self.criterion_consistency = ConsistencyLoss().to(self.device)

    def _calculate_total_loss(self, model_output, batch):
        """Calculates the combined loss based on the config."""
        labels = batch['label'].to(self.device)
        modalities = batch['modality'].to(self.device)
        
        # Supervised prediction losses - using configurable weights
        loss_p1 = self.criterion_ce(model_output['y_predict1'], labels)
        loss_p2 = self.criterion_ce(model_output['y_predict2'], labels)
        
        # Contrastive losses
        loss_itc = self.criterion_itc(model_output['image_projection'], model_output['text_projection'])
        
        p_xi = get_p_xi(labels, modalities, self.class_counts, self.modality_counts)
        loss_iic = self.criterion_iic(
            model_output['image_embedding'], 
            labels, 
            p_xi,
            model_output.get('prototypes', None)
        )
        
        loss_contrast = loss_itc + loss_iic # P(xi) is handled inside IICLoss
        
        # Consistency loss
        loss_consistency = 0
        if config.USE_CONSISTENCY_LOSS:
            loss_consistency = self.criterion_consistency(model_output['y_predict1'], model_output['y_predict2'])
            
                # Total weighted loss - using configurable branch weights
        total_loss = (config.LAMBDA_1 * loss_p1 + 
                     (1 - config.LAMBDA_1) * loss_p2 + 
                     config.LOSS_WEIGHT_CONTRAST * loss_contrast + 
                     config.LOSS_WEIGHT_CONSISTENCY * loss_consistency)
        # total_loss = ((1-config.LAMBDA_1) * loss_p2 +
        #               config.LOSS_WEIGHT_CONTRAST * loss_contrast )
        
        return total_loss

    def _standard_gradient_update(self, batch):
        """Standard gradient update method"""
        self.optimizer.zero_grad()
        model_output = self.model(batch)
        loss = self._calculate_total_loss(model_output, batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_one_epoch(self, epoch, class_groups):
        self.model.train()
        total_loss = 0
        total = 0
        correct = 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        batch_count = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            if config.RUN_SANITY_CHECK and batch_idx >= config.SANITY_CHECK_BATCHES:
                break
            if batch is None: continue
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_count += 1
            
            if config.USE_DYNAMIC_GRADIENT:
                # Dynamic gradient optimization branch
                # Initial grouping or periodic regrouping
                if class_groups is None or batch_count % self.regroup_interval == 0:
                    print(f"Batch {batch_count}: regrouping class groups...")
                    class_groups = gradient_balanced_grouping(self.model, batch, self.metadata['num_classes'], self.n_groups, self.device)
                
                # Main dynamic gradient optimization call with multi-branch loss
                combined_gradient_update(
                    self.model, self.optimizer, batch, class_groups, self.n_groups, self.device,
                    self.normalization_type, self.class_weight, self.sample_weight, loss_fn=self._calculate_total_loss
                )
                
                # Calculate current loss for display
                with torch.no_grad():
                    outputs = self.model(batch)
                    loss = self.criterion_ce(outputs['y_predict1'], batch['label'])
                    total_loss += loss.item()
            else:
                # Standard gradient optimization branch
                loss_value = self._standard_gradient_update(batch)
                total_loss += loss_value
                
            # Calculate accuracy
            with torch.no_grad():
                outputs = self.model(batch)
                # Calculate accuracy with configurable weights
                probs1 = F.softmax(outputs['y_predict1'], dim=1)
                probs2 = F.softmax(outputs['y_predict2'], dim=1)
                probs = config.LAMBDA_1 * probs1 + (1 - config.LAMBDA_1) * probs2
                _, predicted = probs.max(1)
                total += batch['label'].size(0)
                correct += predicted.eq(batch['label']).sum().item()
                current_acc = 100.*correct/total
                
                # Display current loss (CE loss for dynamic gradient, total loss for standard gradient)
                display_loss = loss.item() if config.USE_DYNAMIC_GRADIENT else loss_value
                progress_bar.set_postfix(loss=display_loss, acc=current_acc)
                
                # Check if training accuracy reaches 98%
                if current_acc >= 98.0:
                    print(f"\nTrain accuracy reached 98%! Current accuracy: {current_acc:.2f}%")
                    print("Stopping training and saving model...")
                    return total_loss / (batch_idx+1), current_acc, class_groups, True
                    
        return total_loss / (batch_idx+1), 100.*correct/total, class_groups, False

    def run(self):
        """Main execution function."""
        print(f"\n--- Training for 1 Epoch ---")
        print(f"Branch weights: {config.LAMBDA_1}:{1-config.LAMBDA_1}")
        print(f"Gradient optimization: {'Dynamic' if config.USE_DYNAMIC_GRADIENT else 'Standard'}")
        
        result = self._train_one_epoch(0, None)
        if len(result) == 4:  # Contains early_stop flag
            train_loss, train_acc, class_groups, early_stop = result
            if early_stop:
                print("Saving model due to 98% train accuracy...")
                torch.save(self.model.state_dict(), f"{config.DATASET_NAME}_best_model.pth")
            else:
                print("Training completed. Saving final model...")
                torch.save(self.model.state_dict(), f"{config.DATASET_NAME}_best_model.pth")
        else:  # Normal case
            train_loss, train_acc, class_groups = result
            print("Training completed. Saving final model...")
            torch.save(self.model.state_dict(), f"{config.DATASET_NAME}_best_model.pth")
        
        print(f"Training completed: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.2f}%")
        
        print("\n--- Training Finished. Evaluating on Test Set ---")
        self.model.load_state_dict(torch.load(f"{config.DATASET_NAME}_best_model.pth"))
        self._evaluate_test()
        
    def _evaluate_test(self):
        """Final evaluation on the test set."""
        self.model.eval()
        all_preds, all_labels = [], []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing", leave=False):
                 if batch is None: continue
                 batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                 
                 model_output = self.model(batch)
                 # Fuse two prediction probabilities using configurable weights
                 probs1 = F.softmax(model_output['y_predict1'], dim=1)
                 probs2 = F.softmax(model_output['y_predict2'], dim=1)
                 probs = config.LAMBDA_1 * probs1 + (1 - config.LAMBDA_1) * probs2
                 all_preds.extend(probs.argmax(dim=1).cpu().numpy())
                 all_labels.extend(batch['label'].cpu().numpy())
                 all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels_np = np.array(all_labels)
        n_classes = all_probs.shape[1]
        # Calculate AUC for each class
        aucs = []
        for i in range(n_classes):
            # Need to binarize labels
            binary_labels = (all_labels_np == i).astype(int)
            try:
                auc = roc_auc_score(binary_labels, all_probs[:, i])
            except Exception:
                auc = float('nan')
            aucs.append(auc)
        # macro_AUC
        macro_auc = np.nanmean(aucs)

        # Calculate F1 scores
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print("\n--- Final Test Report ---")
        print(f"Gradient optimization used: {'Dynamic' if config.USE_DYNAMIC_GRADIENT else 'Standard'}")
        print(f"Branch weights used: {config.LAMBDA_1}:{1-config.LAMBDA_1}")
        print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
        print(classification_report(all_labels, all_preds, digits=4))
        # Add confusion matrix printing
        print("Test Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("AUC per class:")
        for i, auc in enumerate(aucs):
            print(f"  Class {i}: AUC = {auc:.4f}")
        print(f"Macro AUC: {macro_auc:.4f}")
        print("F1 per class:")
        for i, f1 in enumerate(f1_per_class):
            print(f"  Class {i}: F1 = {f1:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()