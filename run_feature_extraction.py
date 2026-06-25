import ssl
import certifi
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

_original_create_default_context = ssl.create_default_context

def _patched_create_default_context(*args, **kwargs):
    kwargs.setdefault('cafile', certifi.where())
    return _original_create_default_context(*args, **kwargs)

ssl.create_default_context = _patched_create_default_context
ssl._create_default_https_context = _patched_create_default_context

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_pipeline import CleanedGreekLettersDataset, VGG16FeatureExtractor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}', flush=True)
print(f'Cert bundle: {certifi.where()}', flush=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DATA_DIR = PROJECT_ROOT / 'data' / 'ALPUB_v2' / 'images'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'extracted_features'

dataset = CleanedGreekLettersDataset(root_dir=DATA_DIR, transform=transform)
assert len(dataset) > 0, f'0 images found in {DATA_DIR}'
print(f'Loaded dataset with {len(dataset)} images', flush=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
model = VGG16FeatureExtractor().to(device)
model.eval()

all_features = []
all_labels = []
processed = 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
        processed += len(labels)
        if batch_idx % 20 == 0 or processed == len(dataset):
            print(f'Processed {processed}/{len(dataset)} images', flush=True)

X = np.vstack(all_features)
y = np.concatenate(all_labels)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_DIR / 'X_features_vgg16.npy', X)
np.save(OUTPUT_DIR / 'y_labels.npy', y)

print(f'Saved X to {OUTPUT_DIR / "X_features_vgg16.npy"}', flush=True)
print(f'Saved y to {OUTPUT_DIR / "y_labels.npy"}', flush=True)
print(f'X shape: {X.shape}, y shape: {y.shape}', flush=True)
