import os
import numpy as np
from tqdm import tqdm

# Paths
BASE_PREPROCESSED_DIR = 'data/preprocessed/base/train/'   # or 'test/' for test set
CYCLEGAN_PREPROCESSED_DIR = 'data/preprocessed/cyclegan/train/'  # or 'test/'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_for_cyclegan(npy_path):
    # Load grayscale image (values in [0, 1])
    img = np.load(npy_path)
    # Convert to 3-channel (RGB)
    img_rgb = np.stack([img]*3, axis=-1)
    # Normalize to [-1, 1]
    img_norm = (img_rgb * 2.0) - 1.0
    return img_norm.astype(np.float32)

def main():
    ensure_dir(CYCLEGAN_PREPROCESSED_DIR)
    npy_files = [f for f in os.listdir(BASE_PREPROCESSED_DIR) if f.endswith('.npy')]
    print(f"Found {len(npy_files)} files to process.")

    for fname in tqdm(npy_files, desc="CycleGAN preprocessing"):
        in_path = os.path.join(BASE_PREPROCESSED_DIR, fname)
        out_path = os.path.join(CYCLEGAN_PREPROCESSED_DIR, fname)
        try:
            img_cyclegan = preprocess_for_cyclegan(in_path)
            np.save(out_path, img_cyclegan)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    main()
    print("CycleGAN preprocessing complete!")
