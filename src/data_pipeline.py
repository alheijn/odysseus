import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.models import vgg16, VGG16_Weights
from pathlib import Path
from PIL import Image

# --- 1. Custom Dataset to handle `clean_dataset.py` directory structure ---
class CleanedGreekLettersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Explicitly define valid classes (from your clean_dataset.py)
        self.classes = sorted([
            "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
            "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
            "LunateSigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Traverse and filter out the _flagged_* and _review_* directories
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists(): continue
            
            for img_path in cls_dir.rglob("*.jpg"):
                # Ignore paths that contain flagged directories
                if "_flagged_" in str(img_path) or "_review_" in str(img_path):
                    continue
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# --- 2. VGG-16 Feature Extractor Models ---

# use all 13 blocks
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG-16 weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Retain only the convolutional feature extraction blocks
        self.features = vgg.features
        
        # Apply Global Average Pooling (GAP) to reduce spatial dimensions to 1x1
        # This guarantees an output of exactly 512 dimensions regardless of input size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

# use only the first 3 blocks
class EarlyVGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG-16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Tap into Block 3 (Layer 16 is the final pooling layer of Block 3)
        # This outputs a tensor of shape [Batch, 256, 28, 28]
        self.features = vgg.features[:17] 
        
        # Apply GAP to squash the 28x28 spatial dimensions down to 1x1
        # This yields a flat, 256-dimensional vector focusing on raw strokes and edges
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

class VGG16SPPFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG-16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        
        # --- SPATIAL PYRAMID POOLING (2x2) ---
        # Instead of 1x1, we divide the feature maps into a 2x2 grid.
        # This yields 4 pooled regions per feature map.
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        x = self.features(x)   # Output shape: [Batch, 512, 7, 7]
        x = self.pool(x)       # Output shape: [Batch, 512, 2, 2]
        
        # Flattening 512 channels * 2 * 2 grid yields a 2048-dimensional vector
        return torch.flatten(x, 1) # Output shape: [Batch, 2048]

class VGG16Block4SPPFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG-16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # --- TAP INTO BLOCK 4 ---
        # Layer 23 is the final MaxPool2d of Block 4. 
        # Slicing [:24] keeps layers 0 through 23, discarding Block 5 entirely.
        self.features = vgg.features[:24] 
        
        # --- SPATIAL PYRAMID POOLING (2x2) ---
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        x = self.features(x)   # Output shape (with 256x256 input): [Batch, 512, 16, 16]
        x = self.pool(x)       # Output shape: [Batch, 512, 2, 2]
        
        # Flattening 512 channels * 2 * 2 grid yields a 2048-dimensional vector
        return torch.flatten(x, 1) # Output shape: [Batch, 2048]