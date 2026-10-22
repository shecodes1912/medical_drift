
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from third_party.PraNet.lib.PraNet_Res2Net import PraNet


# Paths
ROOT = Path("data/Kvasir-SEG/Kvasir-SEG")
IMAGE_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"
CHECKPOINT = Path(
    "third_party/PraNet/snapshots/PraNet_Res2Net/PraNet-19.pth"
)
OUT_DIR = Path("outputs/kvasir_calibration")
CSV_PATH = OUT_DIR / "kvasir_scores_all.csv"

# Keep consistent with evaluate_etis.py
IMAGE_SIZE = 352
THRESHOLD = 0.5
SEED = 42
CALIBRATION_FRACTION = 0.80

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def index_images(folder):
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            if path.stem in result:
                raise ValueError(f"Duplicate filename stem: {path.stem}")
            result[path.stem] = path
    return result


def sort_key(stem):
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def entropy_map(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.is_dir() or not MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected image and mask directories:\n"
            f"{IMAGE_DIR}\n{MASK_DIR}"
        )
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    images = index_images(IMAGE_DIR)
    masks = index_images(MASK_DIR)
    paired = sorted(images.keys() & masks.keys(), key=sort_key)

    print(
        f"Images: {len(images)} | Masks: {len(masks)} | "
        f"Paired: {len(paired)}"
    )
    print(f"Images without masks: {len(images.keys() - masks.keys())}")
    print(f"Masks without images: {len(masks.keys() - images.keys())}")

    if not paired:
        raise RuntimeError("No paired Kvasir images found.")

    # Assign the split before inference. Reproducible across runs.
    rng = np.random.default_rng(SEED)
    shuffled_indices = rng.permutation(len(paired))
    n_cal = int(round(CALIBRATION_FRACTION * len(paired)))
    calibration_indices = set(shuffled_indices[:n_cal].tolist())

    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    model = PraNet().to(device)
    state = torch.load(
        CHECKPOINT, map_location=device, weights_only=True
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rows = []

    with torch.inference_mode():
        for i, stem in enumerate(paired):
            image_path = images[stem]
            mask_path = masks[stem]

            image = Image.open(image_path).convert("RGB")
            original_w, original_h = image.size

            resized = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR,
            )
            x = np.asarray(resized, dtype=np.float32) / 255.0
            x = (x - mean) / std
            x = torch.from_numpy(
                x.transpose(2, 0, 1)
            ).unsqueeze(0).to(device)

            logits = model(x)[-1]
            logits = F.interpolate(
                logits,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=True,
            )
            p = torch.sigmoid(logits)[0, 0].cpu().numpy()

            # Label-free monitoring scores
            ent = entropy_map(p)
            pred = p >= THRESHOLD
            low_conf = np.mean((p >= 0.1) & (p <= 0.9))
            _, component_count = ndimage.label(
                pred, structure=np.ones((3, 3), dtype=np.uint8)
            )

            # Ground-truth metrics are recorded separately.
            gt_image = Image.open(mask_path).convert("L")
            if gt_image.size != (original_w, original_h):
                raise ValueError(
                    f"Image/mask size mismatch for {stem}"
                )
            gt = np.asarray(gt_image) > 127

            intersection = np.logical_and(pred, gt).sum()
            pred_sum = pred.sum()
            gt_sum = gt.sum()
            union = np.logical_or(pred, gt).sum()

            dice = (
                2 * intersection / (pred_sum + gt_sum)
                if pred_sum + gt_sum else 1.0
            )
            iou = intersection / union if union else 1.0

            split = (
                "calibration"
                if i in calibration_indices
                else "source_holdout"
            )

            rows.append({
                "image_id": stem,
                "split": split,
                "height": original_h,
                "width": original_w,
                "mean_entropy_nats": float(ent.mean()),
                "p90_pixel_entropy_nats": float(
                    np.quantile(ent, 0.90)
                ),
                "low_confidence_fraction": float(low_conf),
                "connected_component_count": int(component_count),
                "empty_prediction": bool(not pred.any()),
                "pred_foreground_fraction": float(pred.mean()),
                "mean_probability": float(p.mean()),
                "dice": float(dice),
                "iou": float(iou),
            })

            if (i + 1) % 50 == 0 or i + 1 == len(paired):
                print(f"Processed {i + 1}/{len(paired)}")

    fields = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Kvasir scoring complete ===")
    print(f"Total images scored: {len(rows)}")
    print(f"Calibration rows: {sum(r['split'] == 'calibration' for r in rows)}")
    print(f"Held-out source rows: {sum(r['split'] == 'source_holdout' for r in rows)}")
    print(f"Mean Dice (descriptive): {np.mean([r['dice'] for r in rows]):.4f}")
    print(f"Saved: {CSV_PATH}")


if __name__ == "__main__":
    main()
