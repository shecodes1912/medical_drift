import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from third_party.PraNet.lib.PraNet_Res2Net import PraNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an input RGB image")
    parser.add_argument(
        "--checkpoint",
        default="third_party/PraNet/snapshots/PraNet_Res2Net/PraNet-19.pth",
    )
    parser.add_argument("--output-dir", default="outputs/pranet_smoke_test")
    parser.add_argument("--size", type=int, default=352)
    args = parser.parse_args()

    device = torch.device("cpu")
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    original = Image.open(image_path).convert("RGB")
    original_size = original.size

    preprocess = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    tensor = preprocess(original).unsqueeze(0).to(device)

    model = PraNet().to(device)
    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.inference_mode():
        outputs = model(tensor)
        logits = outputs[-1]
        logits = F.interpolate(
            logits,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Save probabilities directly; do not min-max normalize them.
    prob_path = output_dir / f"{image_path.stem}_probability.png"
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    np.save(output_dir / f"{image_path.stem}_probability.npy", probabilities)

    Image.fromarray(
        np.clip(probabilities * 255, 0, 255).astype(np.uint8)
    ).save(prob_path)

    binary_mask = (probabilities >= 0.5).astype(np.uint8) * 255
    Image.fromarray(binary_mask).save(mask_path)

    eps = 1e-7
    p = np.clip(probabilities, eps, 1 - eps)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    mean_entropy = float(entropy.mean())

    print(f"Device: {device}")
    print(f"Input: {image_path} | original size: {original_size}")
    print(f"Probability range: {probabilities.min():.6f} to {probabilities.max():.6f}")
    print(f"Mean pixel entropy (nats): {mean_entropy:.6f}")
    print(f"Saved probability preview: {prob_path}")
    print(f"Saved binary mask: {mask_path}")
    print(f"Saved full-precision probabilities: {output_dir / (image_path.stem + '_probability.npy')}")


if __name__ == "__main__":
    main()
