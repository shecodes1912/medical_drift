import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from config import cfg

class FlexibleMedicalDataset(Dataset):
    """
    Robust Dataset Loader supporting multiple folder conventions 
    (images/masks, PNG/Original, Ground Truth, TIF, nested subfolders).
    """
    def __init__(self, root_dir: str, img_size: tuple = cfg.img_size):
        self.root_dir = root_dir
        self.img_size = img_size
        
        self.img_paths = []
        self.mask_paths = []
        
        # 1. Check for standard 'images' & 'masks'
        img_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")
        
        # 2. Check for CVC layout: 'PNG/Original' and 'PNG/Ground Truth'
        if not os.path.exists(img_dir):
            cvc_img_dir = os.path.join(root_dir, "PNG", "Original")
            cvc_mask_dir = os.path.join(root_dir, "PNG", "Ground Truth")
            if os.path.exists(cvc_img_dir):
                img_dir, mask_dir = cvc_img_dir, cvc_mask_dir
                
        # 3. Check for MONU-SEG split folders (training/test)
        if not os.path.exists(img_dir):
            for sub in ["kmms_training/kmms_training", "kmms_test/kmms_test"]:
                sub_img = os.path.join(root_dir, sub, "images")
                sub_mask = os.path.join(root_dir, sub, "masks")
                if os.path.exists(sub_img):
                    img_dir, mask_dir = sub_img, sub_mask
                    break

        # Discover all valid image files
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.PNG", "*.JPG", "*.TIF"]:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, "**", ext), recursive=True))
            self.mask_paths.extend(glob.glob(os.path.join(mask_dir, "**", ext), recursive=True))
            
        self.img_paths = sorted(self.img_paths)
        self.mask_paths = sorted(self.mask_paths)
        
        # Verify alignment
        print(f"Loaded dataset from '{root_dir}': {len(self.img_paths)} images, {len(self.mask_paths)} masks.")
        
        # Transforms
        self.img_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.img_transform(image)
        
        sample = {"image": img_tensor, "path": img_path}
        
        if idx < len(self.mask_paths):
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
            sample["mask"] = (mask_tensor > 0.5).float()
            
        return sample


def get_dataloader(data_dir: str, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    dataset = FlexibleMedicalDataset(root_dir=data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

if __name__ == "__main__":
    from config import cfg
    
    print("--- Testing Dataset Loaders ---")
    
    # 1. Kvasir-SEG
    kvasir_loader = get_dataloader(cfg.kvasir_dir)
    
    # 2. CVC-ClinicDB
    cvc_loader = get_dataloader(cfg.cvc_dir)
    
    # 3. ETIS-LaribPolypDB
    etis_loader = get_dataloader(cfg.etis_dir)
    
    # 4. MONU-SEG
    monuseg_loader = get_dataloader(cfg.monuseg_dir)