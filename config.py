import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Experiment Hyperparameters
    delta: float = 0.05            # Permissible false alarm rate (Ville's threshold = 20)
    epsilon: float = 0.5           # Power martingale parameter
    cal_sizes: List[int] = field(default_factory=lambda: [10, 25, 50, 100, 200])
    default_n_cal: int = 50
    num_runs: int = 500
    
    # Image & Score Parameters
    img_size: tuple = (256, 256)
    prediction_set_lambda: float = 0.9
    
    # Dataset Paths matching your local structure
    data_dir: str = "./data"
    kvasir_dir: str = os.path.join(data_dir, "Kvasir-SEG", "Kvasir-SEG")
    cvc_dir: str = os.path.join(data_dir, "CVC_Clinical data")
    etis_dir: str = os.path.join(data_dir, "ETIS-LaribPolypDB")
    monuseg_dir: str = os.path.join(data_dir, "MONU-SEG")
    
    # Outputs
    output_dir: str = "./results"

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)