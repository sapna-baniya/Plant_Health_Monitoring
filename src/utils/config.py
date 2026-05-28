import yaml
import torch

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    device = cfg["training"].get("device", "auto")
    if device == "auto":
        cfg["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg
