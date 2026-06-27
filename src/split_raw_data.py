import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def physical_train_test_split(raw_data_dir, output_dir, test_size=0.20):
    root = Path(raw_data_dir)
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"
    
    classes = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('_')]
    
    for cls_dir in tqdm(classes, desc="Splitting Classes"):
        # Create output directories
        (train_dir / cls_dir.name).mkdir(parents=True, exist_ok=True)
        (test_dir / cls_dir.name).mkdir(parents=True, exist_ok=True)
        
        # Get all images for this class
        images = list(cls_dir.glob("*.jpg"))
        random.shuffle(images) # Shuffle for randomness
        
        # Calculate split index
        split_idx = int(len(images) * (1 - test_size))
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        # Copy to respective folders
        for img in train_imgs:
            shutil.copy(img, train_dir / cls_dir.name / img.name)
        for img in test_imgs:
            shutil.copy(img, test_dir / cls_dir.name / img.name)
            
    print(f"Split complete! Data saved to {train_dir} and {test_dir}")

if __name__ == "__main__":
    # Adjust paths based on your actual raw dataset location
    physical_train_test_split(
        raw_data_dir="../data/ALPUB_v2/images", 
        output_dir="../data/split_dataset"
    )