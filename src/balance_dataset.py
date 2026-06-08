import os
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def balance_dataset_offline(data_dir):
    """
    This script calculates the size of the largest class and artificially generates modified images for all other classes until every letter has the exact same amount of data.
    """
    root = Path(data_dir)
    classes = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith('_')]
    
    # 1. Find the target number of images (the size of the largest class)
    class_counts = {c.name: len(list(c.glob("*.jpg"))) for c in classes}
    target_count = max(class_counts.values())
    
    print(f"Target count per class: {target_count} images")
    
    # 2. Define the geometric augmentations
    # These simulate bad cropping, angled papyrus, and minor scaling
    augmenter = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,          # Slight rotations
            translate=(0.1, 0.1),# Shift the letter up/down/left/right by 10%
            scale=(0.9, 1.1),    # Slight zoom in/out
            fill=180             # Fills empty space with a generic papyrus-like gray/ochre value
        )
    ])
    
    # 3. Augment minority classes
    for cls_dir in classes:
        images = list(cls_dir.glob("*.jpg"))
        current_count = len(images)
        deficit = target_count - current_count
        
        if deficit <= 0:
            print(f"Skipping {cls_dir.name} (already at {current_count})")
            continue
            
        print(f"Augmenting {cls_dir.name}: Generating {deficit} new images...")
        
        # Randomly sample existing images to augment
        for i in tqdm(range(deficit), leave=False):
            img_path = random.choice(images)
            try:
                img = Image.open(img_path).convert("RGB")
                aug_img = augmenter(img)
                # Save with a prefix to indicate it's synthetic
                save_path = cls_dir / f"aug_{i}_{img_path.name}"
                aug_img.save(save_path)
            except Exception as e:
                continue

if __name__ == "__main__":
    # Adjust path to your clean dataset folder
    balance_dataset_offline("../data/ALPUB_v2/images")