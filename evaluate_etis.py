from pathlib import Path
import csv
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy import ndimage

from third_party.PraNet.lib.PraNet_Res2Net import PraNet

ROOT = Path("data/ETIS-LaribPolypDB")
IMAGE_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"
CHECKPOINT = Path("third_party/PraNet/snapshots/PraNet_Res2Net/PraNet-19.pth")
OUT_DIR = Path("outputs/etis_evaluation")
CSV_PATH = OUT_DIR / "etis_sample50.csv"
SAMPLE_SIZE = 196
IMAGE_SIZE = 352
THRESHOLD = 0.5

OUT_DIR.mkdir(parents=True, exist_ok=True)

def index_images(folder):
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in supported:
            if path.stem in result:
                raise ValueError(f"Duplicate filename stem in {folder}: {path.stem}")
            result[path.stem] = path
    return result

images = index_images(IMAGE_DIR)
masks = index_images(MASK_DIR)
common = sorted(images.keys() & masks.keys(), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
missing_masks = sorted(images.keys() - masks.keys())
missing_images = sorted(masks.keys() - images.keys())

print(f"Images: {len(images)} | Masks: {len(masks)} | Paired: {len(common)}")
print(f"Images without matching masks: {len(missing_masks)}")
print(f"Masks without matching images: {len(missing_images)}")

if not common:
    raise RuntimeError("No image/mask pairs found.")
if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

# Select evenly across the sorted paired dataset, deterministically.
indices = np.linspace(0, len(common) - 1, min(SAMPLE_SIZE, len(common)), dtype=int)
selected = [common[i] for i in indices]

device = torch.device("cpu")
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
model = PraNet().to(device)

state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
rows = []
    
def entropy_map(p):
    # Convert float32 probabilities to float64 BEFORE clipping.
    # Otherwise, 1 - 1e-12 can round back to 1.0 in float32.
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


with torch.inference_mode():
    for number, stem in enumerate(selected, 1):
        image_path = images[stem]
        mask_path = masks[stem]

        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size
        resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        x = np.asarray(resized, dtype=np.float32) / 255.0
        x = (x - mean) / std
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

        outputs = model(x)
        logits = outputs[-1]
        logits = F.interpolate(
            logits, size=(original_h, original_w),
            mode="bilinear", align_corners=True
        )
        p = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = p >= THRESHOLD
        # Label-free signals: computed from predictions only.
        low_confidence_fraction = float(
            np.mean((p >= 0.1) & (p <= 0.9))
        )
        empty_prediction = bool(not pred.any())

        # Count 8-connected components in the predicted foreground.
        # An empty prediction has zero components and is flagged separately.
        _, component_count = ndimage.label(
            pred, structure=np.ones((3, 3), dtype=np.uint8)
        )

        gt_image = Image.open(mask_path).convert("L")
        if gt_image.size != (original_w, original_h):
            raise ValueError(
                f"Image/mask size mismatch for {stem}: "
                f"image={(original_w, original_h)}, mask={gt_image.size}"
            )
        gt = np.asarray(gt_image) > 127

        intersection = np.logical_and(pred, gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()
        union = np.logical_or(pred, gt).sum()

        dice = (2 * intersection / (pred_sum + gt_sum)
                if pred_sum + gt_sum else 1.0)
        iou = intersection / union if union else 1.0
        ent = entropy_map(p)

        rows.append({
            "image_id": stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "height": original_h,
            "width": original_w,
            "dice": float(dice),
            "iou": float(iou),
            "mean_entropy_nats": float(ent.mean()),
            "low_confidence_fraction": low_confidence_fraction,
            "connected_component_count": int(component_count),
            "empty_prediction": empty_prediction,
            "p90_pixel_entropy_nats": float(np.quantile(ent, 0.90)),
            "pred_foreground_fraction": float(pred.mean()),
            "gt_foreground_fraction": float(gt.mean()),
            "mean_probability": float(p.mean()),
        })

        if number == 1:
            np.save(OUT_DIR / f"{stem}_probability.npy", p)
        if number % 10 == 0 or number == len(selected):
            print(f"Processed {number}/{len(selected)}")

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

def rank_values(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks

def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

dice_values = [r["dice"] for r in rows]
iou_values = [r["iou"] for r in rows]
entropy_values = [r["mean_entropy_nats"] for r in rows]

print("\n=== ETIS sample evaluation ===")
print(f"Images evaluated: {len(rows)}")
print(f"Mean Dice:   {np.mean(dice_values):.4f} +/- {np.std(dice_values, ddof=1):.4f}")
print(f"Median Dice: {np.median(dice_values):.4f}")
print(f"Mean IoU:    {np.mean(iou_values):.4f} +/- {np.std(iou_values, ddof=1):.4f}")
print(f"Mean entropy (nats): {np.mean(entropy_values):.6f}")
print("Pearson correlation: entropy vs Dice = "
      f"{correlation(entropy_values, dice_values):.4f}")
print("Spearman correlation: entropy vs Dice = "
      f"{correlation(rank_values(entropy_values), rank_values(dice_values)):.4f}")
print(f"\nPer-image results saved to: {CSV_PATH}")
