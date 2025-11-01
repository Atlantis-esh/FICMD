import torch

# --- General Settings ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/devdata/PUBLIC_DATASET/Dataset2_Tumor"
CAPTIONS_FILE = "/devdata/PUBLIC_DATASET/Dataset2_Tumor/Final_labels.csv"
DATASET_NAME = "AD"

BATCH_SIZE = 32

# --- Loss Weights ---
# Final Loss = λ1*L_y_predict1 + (1-λ1)*L_y_predict2 + 
#              λ_con*L_contrast + λ_rec*L_consistency
# Based on paper's loss formulation in Equation 14
LAMBDA_1 = 0.7  # Weight for PDE branch vs MEE branch (optimal from paper)
LOSS_WEIGHT_CONTRAST = 0.3  # λ_con - Contrastive loss weight
LOSS_WEIGHT_CONSISTENCY = 0.8   # λ_rec - Consistency regularization weight

# --- Gradient Optimization Settings ---
USE_DYNAMIC_GRADIENT = False  # Set to False for standard gradient optimization
RANDOM_SEED = 114514
DATA_TRAIN_RATIO = 0.8

NUM_WORKERS = 4
PIN_MEMORY = True

# --- Data Processing ---
IMG_SIZE = 224
TRAIN_PCT = 0.7
VAL_PCT = 0.15
# TEST_PCT is implicitly 1 - TRAIN_PCT - VAL_PCT

# --- Model Architecture ---
IMAGE_ENCODER = 'resnet50'  # From timm library
TEXT_ENCODER = 'distilbert-base-uncased'  # From transformers
EMBEDDING_DIM = 512  # Internal embedding dimension for many modules
NUM_CROSS_ATTENTION_HEADS = 8

# --- Training Hyperparameters ---
EPOCHS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# --- PDE Module Settings ---
# Switch to enable/disable modality-based weighting in the cross-attention module.
USE_MODALITY_WEIGHTS = True

# P(xi) can be 0, 1, or 'dynamic'
# 'dynamic': P(xi) is calculated based on class and modality rarity.
# 1: P(xi) = 1, so L_contrast = L_ITC + L_IIC.
# 0: P(xi) = 0, so L_contrast = L_ITC.
P_XI_FACTOR = 'dynamic'  # Options: 'dynamic', 1, 0

# Switch to enable/disable the prototype mechanism in the IIC loss.
USE_IIC_PROTOTYPES = True
PROTOTYPE_UPDATE_ALPHA = 0.99  # Momentum for prototype updates

# --- ASGO Module Settings ---
GRADIENT_NORMALIZATION_TYPE = 'l2'
REGROUP_INTERVAL = 100  # How often to re-run spectral clustering
NUM_GRADIENT_GROUPS = 3  # Number of groups for clustering
W_MIX = 0.4  # Weight for sample-level vs class-level gradients (optimal from paper)

# --- Consistency Regularization ---
USE_CONSISTENCY_LOSS = True

# --- Early Stopping ---
EARLY_STOP_PATIENCE = 10

# --- Debugging ---
RUN_SANITY_CHECK = False
SANITY_CHECK_BATCHES = 5 

