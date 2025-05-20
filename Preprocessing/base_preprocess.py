import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ================== CONFIGURATION ==================
CSV_PATH = 'data/NIH_Resized_dataset/Data_Entry_2017.csv'
IMAGE_DIR = 'data/NIH_Resized_dataset/images-224/images-224'
OUTPUT_DIR = 'data/preprocessed/base/'
IMG_SIZE = (224, 224)
TEST_RATIO = 0.2
# ====================================================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def verify_paths():
    """Check critical paths exist before processing"""
    print("\n=== Path Verification ===")
    print(f"CSV file exists: {os.path.exists(CSV_PATH)}")
    print(f"Image directory exists: {os.path.exists(IMAGE_DIR)}")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found at {IMAGE_DIR}")

def preprocess_image(img_path):
    """Load and preprocess single image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32) / 255.0  # Normalize to [0,1]

def main():
    verify_paths()  # First check critical paths
    
    # Load metadata
    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} entries from CSV")

    # Create image paths
    df['img_path'] = df['Image Index'].apply(lambda x: os.path.join(IMAGE_DIR, x))
    
    # Filter valid paths
    df['exists'] = df['img_path'].apply(os.path.exists)
    print(f"Found {df['exists'].sum()} valid images out of {len(df)}")
    
    if df['exists'].sum() == 0:
        sample_path = df['img_path'].iloc[0]
        raise FileNotFoundError(
            f"No valid images found. Example path: {sample_path}\n"
            f"Check CSV column names and IMAGE_DIR path."
        )
    
    df = df[df['exists']].drop(columns=['exists'])
    
    # Train/test split
    train_df, test_df = train_test_split(df, test_size=TEST_RATIO, random_state=42)
    print(f"\nSplit sizes - Train: {len(train_df)}, Test: {len(test_df)}")

    # Create output directories
    ensure_dir(os.path.join(OUTPUT_DIR, 'train'))
    ensure_dir(os.path.join(OUTPUT_DIR, 'test'))

    # Save metadata
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_metadata.csv'), index=False)

    # Process and save images
    for split, split_df in [('train', train_df), ('test', test_df)]:
        print(f"\nProcessing {split} set...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            try:
                img = preprocess_image(row['img_path'])
                output_path = os.path.join(OUTPUT_DIR, split, row['Image Index'].replace('.png', '.npy'))
                np.save(output_path, img)
            except Exception as e:
                print(f"\nError processing {row['Image Index']}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
    print("\nPreprocessing completed successfully!")
