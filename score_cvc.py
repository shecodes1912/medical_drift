
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from third_party.PraNet.lib.PraNet_Res2Net import PraNet

ROOT = Path("data/CVC_Clinical data")
METADATA = ROOT / "metadata.csv"
CHECKPOINT = Path(
    "third_party/PraNet/snapshots/PraNet_Res2Net/PraNet-19.pth"
)
OUT_DIR = Path("outputs/cvc_evaluation")
CSV_PATH = OUT_DIR / "cvc_scores_all.csv"

IMAGE_SIZE = 352
THRESHOLD = 0.5


def entropy_map(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (METADATA, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)

    with METADATA.open("r", newline="", encoding="utf-8-sig") as f:
        metadata = list(csv.DictReader(f))

    required = {"frame_id", "png_image_path", "png_mask_path"}
    if not metadata or not required.issubset(metadata[0]):
        raise ValueError(
            f"Metadata must contain columns: {sorted(required)}"
        )

    pairs = []
    seen = set()

    for row in metadata:
        image_path = ROOT / row["png_image_path"]
        mask_path = ROOT / row["png_mask_path"]

        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing pair for frame {row['frame_id']}: "
                f"{image_path} | {mask_path}"
            )

        key = row["frame_id"]
        if key in seen:
            raise ValueError(f"Duplicate frame_id: {key}")
        seen.add(key)
        pairs.append((row, image_path, mask_path))

    print(f"Metadata rows: {len(metadata)}")
    print(f"Validated PNG image-mask pairs: {len(pairs)}")

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
        for i, (meta, image_path, mask_path) in enumerate(pairs, 1):
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

            # Label-free scores: no ground truth used here.
            ent = entropy_map(p)
            pred = p >= THRESHOLD
            _, component_count = ndimage.label(
                pred, structure=np.ones((3, 3), dtype=np.uint8)
            )

            low_conf = np.mean((p >= 0.1) & (p <= 0.9))

            # Ground truth is used only for retrospective evaluation.
            mask_image = Image.open(mask_path).convert("L")
            if mask_image.size != (original_w, original_h):
                raise ValueError(
                    f"Image/mask size mismatch: {image_path.name}"
                )
            gt = np.asarray(mask_image) > 127

            intersection = np.logical_and(pred, gt).sum()
            pred_sum = pred.sum()
            gt_sum = gt.sum()
            union = np.logical_or(pred, gt).sum()

            dice = (
                2 * intersection / (pred_sum + gt_sum)
                if pred_sum + gt_sum else 1.0
            )
            iou = intersection / union if union else 1.0

            rows.append({
                "image_id": meta["frame_id"],
                "sequence_id": meta.get("sequence_id", ""),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
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

            if i % 25 == 0 or i == len(pairs):
                print(f"Processed {i}/{len(pairs)}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== CVC scoring complete ===")
    print(f"Images evaluated: {len(rows)}")
    print(f"Mean Dice: {np.mean([r['dice'] for r in rows]):.4f}")
    print(f"Median Dice: {np.median([r['dice'] for r in rows]):.4f}")
    print(f"Mean IoU: {np.mean([r['iou'] for r in rows]):.4f}")
    print(f"Empty predictions: {sum(r['empty_prediction'] for r in rows)}")
    print(f"Saved: {CSV_PATH}")


if __name__ == "__main__":
    main()
