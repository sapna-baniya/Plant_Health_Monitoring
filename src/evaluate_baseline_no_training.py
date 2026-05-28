"""Baseline evaluation after adapting output heads but before fine-tuning.
This satisfies the project requirement to compare pre-finetuning and post-finetuning behavior.
"""
import argparse
import os
import torch
from dataset_coco import PlantCocoDataset, collate_fn
from models.maskrcnn_model import get_maskrcnn_model
from utils.config import load_config
from evaluate_maskrcnn import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"])
    ds = PlantCocoDataset(cfg["data"][f"{args.split}_images"], cfg["data"][f"{args.split}_annotations"], cfg["training"]["image_size"])
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=cfg["training"].get("num_workers", 0), collate_fn=collate_fn)
    model = get_maskrcnn_model(cfg["num_classes"]).to(device)
    # The backbone starts from COCO-pretrained weights, but the output heads are newly initialized.
    # No plant-data fine-tuning is performed here.
    os.makedirs("outputs/metrics", exist_ok=True)
    results = evaluate(model, loader, device, args.threshold, f"outputs/metrics/baseline_no_training_{args.split}.csv")
    with open("outputs/metrics/baseline_no_training_summary.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
