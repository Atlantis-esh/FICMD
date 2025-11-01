import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    ViTModel, ViTImageProcessor
)

# === Configuration ===
BIOGPT_PATH = "/devdata/models/biogpt/"  # Can be replaced with BioGPT-Lite
VIT_PATH = "/devdata/models/vit-base-patch16-224/"
IMAGE_FOLDER ="/devdata/PUBLIC_DATASET/Dataset2_Tumor/images/"
MODALITY_CSV = "/devdata/PUBLIC_DATASET/Dataset2_Tumor/labels.csv"
OUTPUT_CSV = "/devdata/PUBLIC_DATASET/Dataset2_Tumor/image_caption.csv"
BATCH_SIZE = 16
MAX_TOKENS = 50

# === Device detection ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Preload modality information ===
print("Loading modality information...")
modality_df = pd.read_csv(MODALITY_CSV)
modality_dict = dict(zip(modality_df['image'], modality_df['modality']))
print(f"Loaded modality information for {len(modality_dict)} images")

# === Model loading ===
tokenizer = AutoTokenizer.from_pretrained(BIOGPT_PATH)
tokenizer.padding_side = 'left'  # Prevent decoder-only padding misalignment
tokenizer.model_max_length = 512  # Set maximum length to avoid truncation warning

text_model = AutoModelForCausalLM.from_pretrained(BIOGPT_PATH).to(device)
text_model.eval()

image_model = ViTModel.from_pretrained(VIT_PATH).to(device)
image_model.eval()
image_processor = ViTImageProcessor.from_pretrained(VIT_PATH)

# === Image batch feature extraction ===
def extract_features(fnames):
    images, valid_fnames = [], []
    for fname in fnames:
        try:
            img = Image.open(os.path.join(IMAGE_FOLDER, fname)).convert("RGB")
            images.append(img)
            valid_fnames.append(fname)
        except Exception as e:
            print(f"[ERROR] Cannot open {fname}: {e}")
    if not images:
        return []

    inputs = image_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = image_model(**inputs)
        pooled_feats = outputs.pooler_output.mean(dim=1).tolist()

    prompts = []
    for fname, pooled_feat in zip(valid_fnames, pooled_feats):
        modality = modality_dict.get(fname, "Unknown")
        prompt = f"This is a medical {modality} image. Average embedding value is {round(pooled_feat, 3)}. Findings:"
        prompts.append(prompt)

    return list(zip(valid_fnames, prompts))

# === Main function ===
def process_all_images():
    image_list = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    batches = [image_list[i:i + BATCH_SIZE] for i in range(0, len(image_list), BATCH_SIZE)]
    print(f"Total images: {len(image_list)}, Batch size: {BATCH_SIZE}, Total batches: {len(batches)}")

    all_results = []
    for batch in tqdm(batches):
        prompt_pairs = extract_features(batch)
        if not prompt_pairs:
            continue

        prompts = [p for _, p in prompt_pairs]
        fnames = [f for f, _ in prompt_pairs]

        try:
            input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            with torch.no_grad():
                outputs = text_model.generate(
                    input_ids,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8
                )
            captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for fname, caption in zip(fnames, captions):
                all_results.append({"image": fname, "caption": caption})
        except Exception as e:
            print(f"[ERROR] Batch failed: {e}")
            for fname in fnames:
                all_results.append({"image": fname, "caption": f"ERROR: {e}"})

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Captions saved to {OUTPUT_CSV}")



if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    import time
    start = time.time()
    process_all_images()
    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")