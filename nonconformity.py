import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

def compute_entropy_s1(probs: torch.Tensor) -> float:
    """
    S1: Per-image normalized spatial entropy.
    probs: Model output probabilities of shape (C, H, W) or (H, W).
    """
    if probs.ndim == 2:
        # Convert binary probability to 2-class softmax
        p1 = probs
        p0 = 1.0 - probs
        probs = torch.stack([p0, p1], dim=0)
    
    # Clamp to avoid log(0)
    probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
    
    # Per-pixel entropy: -sum(p * log(p)) over classes
    pixel_entropy = -torch.sum(probs * torch.log(probs), dim=0)
    
    # Return average entropy across all pixels
    return pixel_entropy.mean().item()


def compute_prediction_set_size_s2(probs: torch.Tensor, threshold: float = 0.9) -> float:
    """
    S2: Proportion of ambiguous pixels where maximum confidence is below threshold.
    """
    if probs.ndim == 2:
        max_conf = torch.max(probs, 1.0 - probs)
    else:
        max_conf, _ = torch.max(probs, dim=0)
    
    ambiguous_pixels = (max_conf < threshold).float()
    return ambiguous_pixels.mean().item()


def compute_fragmentation_s3(pred_mask: np.ndarray) -> float:
    """
    S3: Topological nonconformity - count of disconnected predicted components.
    pred_mask: Binary mask of shape (H, W).
    """
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    labeled_array, num_features = label(binary_mask)
    return float(num_features)


def extract_nonconformity_score(probs: torch.Tensor, score_type: str = "S1") -> float:
    """Unified score extractor."""
    if score_type == "S1":
        return compute_entropy_s1(probs)
    elif score_type == "S2":
        return compute_prediction_set_size_s2(probs)
    elif score_type == "S3":
        pred_mask = (probs > 0.5).detach().cpu().numpy().squeeze()
        return compute_fragmentation_s3(pred_mask)
    else:
        raise ValueError(f"Unknown score type: {score_type}")