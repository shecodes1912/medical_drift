import torch
import torch.nn as nn
import torch.nn.functional as F

class FrozenSegmenterWrapper:
    """
    Base wrapper for frozen segmentation models.
    Guarantees zero weight updates and pure output probability generation.
    """
    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs forward pass and returns per-pixel output probabilities p in [0, 1].
        Output shape: (B, C, H, W) or (B, H, W).
        """
        x = x.to(self.device)
        logits = self.model(x)
        
        if isinstance(logits, (tuple, list)):
            logits = logits[0]  # Handle multi-output models like PraNet
            
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)
            
        return probs.squeeze(0).cpu()  # Return single image tensor (C, H, W) or (H, W)


class BaselineDummySegmenter(nn.Module):
    """
    Lightweight mock segmenter for pipeline integration testing and offline verification.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv(x)
