#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import uniform_filter
import shutil

# ─── SETTINGS ─────────────────────────────────────────────────────────────────

# set this to your dataset folder
ROOT = Path(input("Enter path to dataset folder: "))

LETTERS = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
    "LunateSigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"
]

# letters with 10000+ samples — aggressive filtering
RADICAL = {"Alpha", "Epsilon", "Iota", "Nu", "Omicron", "Rho", "LunateSigma", "Tau"}
# letters with ~300 samples — keep as much as possible
SAFE    = {"Beta", "Zeta", "Xi", "Psi"}
# everything else — middle ground

# brightness threshold for white blob detection
WHITE_BLOB_BRIGHTNESS = 160
# minimum connected bright blob size (% of image) to flag, per strategy
WHITE_BLOB_PCT = {
    "radical": 0.05,
    "safe":    3.0,
    "middle":  1.5,
}

# images with mean brightness below this are considered too dark
DARK_THRESHOLD = 55
# images with local contrast below this are considered to have no visible symbol
CONTRAST_THRESHOLD = 6.5
# gray blob larger than this % → move to flagged
GRAY_BLOB_CERTAIN = 3.5
# gray blob larger than this % → move to review (borderline cases)
GRAY_BLOB_BORDERLINE = 1.2

# ─── FOUR DETECTION FUNCTIONS ─────────────────────────────────────────────────

def is_white_blob(path, blob_pct_thr):
    # detects overexposed bright spots (white or bright ochre)
    # finds the largest connected region of pixels above brightness threshold
    # and checks if it exceeds the allowed % of image area
    img = Image.open(path).convert("RGB")
    brightness = np.array(img).mean(axis=2)
    mask = brightness >= WHITE_BLOB_BRIGHTNESS
    if not mask.any():
        return False
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return 100.0 * max(sizes) / mask.size >= blob_pct_thr

def is_too_dark(path):
    # detects images that are too dark overall (scanning artifacts, underexposure)
    # uses mean brightness across all pixels and all RGB channels
    img = Image.open(path).convert("RGB")
    return np.array(img).mean() < DARK_THRESHOLD

def is_low_contrast(path):
    # detects images where the symbol is absent or invisible
    # measures average difference between each pixel and its 7x7 neighborhood mean
    # low value = no sharp edges = no ink strokes
    img = Image.open(path).convert("RGB")
    b = np.array(img).mean(axis=2)
    local_mean = uniform_filter(b, size=7)
    return np.abs(b - local_mean).mean() < CONTRAST_THRESHOLD

def gray_blob_pct(path):
    # detects gray foreign objects (paper, tape, card) on the ochre background
    # ochre has high R-B difference; gray has R≈B (low R-B difference)
    # combines low saturation (R-B < 20) and sufficient brightness (not dark ink)
    # returns the size of the largest such connected region as % of image area
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    R = arr[:,:,0].astype(float)
    B = arr[:,:,2].astype(float)
    brightness = arr.mean(axis=2)
    mask = (R - B < 20) & (brightness > 60)
    if not mask.any():
        return 0.0
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return 100.0 * max(sizes) / mask.size

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

def get_strategy(letter):
    if letter in RADICAL: return "radical"
    if letter in SAFE:    return "safe"
    return "middle"


total_stats = {"total": 0, "white": 0, "dark": 0, "contrast": 0,
               "gray": 0, "review": 0, "clean": 0}

for letter in LETTERS:
    folder = ROOT / letter
    if not folder.exists():
        print(f"  {letter}: folder not found, skipping")
        continue

    print(f" Starting for letter: {letter}")

    strategy = get_strategy(letter)
    blob_thr = WHITE_BLOB_PCT[strategy]

    # create output folders for flagged images
    dir_white    = folder / "_flagged_white_blob"
    dir_dark     = folder / "_flagged_too_dark"
    dir_contrast = folder / "_flagged_low_contrast"
    dir_gray     = folder / "_flagged_gray_blob"
    dir_review   = folder / "_review_gray_blob"
    for d in [dir_white, dir_dark, dir_contrast, dir_gray, dir_review]:
        d.mkdir(exist_ok=True)

    images = list(folder.glob("*.jpg"))
    w = dk = lc = gf = gr = 0
    remaining = images

    # step 1 — remove white/bright blobs
    next_remaining = []
    for p in remaining:
        if is_white_blob(p, blob_thr):
            shutil.move(str(p), dir_white / p.name)
            w += 1
        else:
            next_remaining.append(p)
    remaining = next_remaining

    # step 2 — remove too dark images
    next_remaining = []
    for p in remaining:
        if is_too_dark(p):
            shutil.move(str(p), dir_dark / p.name)
            dk += 1
        else:
            next_remaining.append(p)
    remaining = next_remaining

    # step 3 — remove images with no visible symbol (low contrast)
    next_remaining = []
    for p in remaining:
        if is_low_contrast(p):
            shutil.move(str(p), dir_contrast / p.name)
            lc += 1
        else:
            next_remaining.append(p)
    remaining = next_remaining

    # step 4 — remove gray blobs; borderline cases go to review folder
    next_remaining = []
    for p in remaining:
        pct = gray_blob_pct(p)
        if pct >= GRAY_BLOB_CERTAIN:
            shutil.move(str(p), dir_gray / p.name)
            gf += 1
        elif pct >= GRAY_BLOB_BORDERLINE:
            shutil.move(str(p), dir_review / p.name)
            gr += 1
        else:
            next_remaining.append(p)
    remaining = next_remaining

    clean = len(remaining)
    total_stats["total"]    += len(images)
    total_stats["white"]    += w
    total_stats["dark"]     += dk
    total_stats["contrast"] += lc
    total_stats["gray"]     += gf
    total_stats["review"]   += gr
    total_stats["clean"]    += clean

    print(f"{letter:<15} [{strategy:<7}]  total={len(images):>5}  "
          f"white={w:>4}  dark={dk:>4}  contrast={lc:>4}  "
          f"gray={gf:>4}  review={gr:>4}  clean={clean:>5}")

print("\n" + "─" * 90)
print(f"{'TOTAL':<26}"
      f"  total={total_stats['total']:>5}  "
      f"white={total_stats['white']:>4}  "
      f"dark={total_stats['dark']:>4}  "
      f"contrast={total_stats['contrast']:>4}  "
      f"gray={total_stats['gray']:>4}  "
      f"review={total_stats['review']:>4}  "
      f"clean={total_stats['clean']:>5}")





