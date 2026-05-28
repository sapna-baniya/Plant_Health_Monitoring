import argparse
import csv
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from dataset_coco import PlantCocoDataset, collate_fn
from models.maskrcnn_model import get_maskrcnn_model
from utils.config import load_config
from utils.metrics import mask_iou, dice_score


@torch.no_grad()
def evaluate(model, loader, device, score_threshold=0.05, output_csv=None):
    model.eval()
    ious, dices, inference_times = [], [], []
    image_rows = []

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Evaluating")):
        images = [img.to(device) for img in images]
        start = time.time()
        outputs = model(images)
        inference_times.append((time.time() - start) / len(images))

        for output, target in zip(outputs, targets):
            pred_masks = output["masks"].detach().cpu()
            pred_scores = output["scores"].detach().cpu()
            pred_labels = output["labels"].detach().cpu()
            true_masks = target["masks"].detach().cpu()
            true_labels = target["labels"].detach().cpu()

            keep = pred_scores >= score_threshold
            pred_masks = pred_masks[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            per_img_ious, per_img_dices = [], []
            for i in range(len(true_masks)):
                true_mask = true_masks[i]
                true_label = true_labels[i]

                # Prefer same-class match; if none, record zero for that object.
                class_matches = (pred_labels == true_label).nonzero().flatten()
                best_iou, best_dice = 0.0, 0.0
                for idx in class_matches:
                    pmask = pred_masks[idx, 0] > 0.5
                    iou = mask_iou(pmask, true_mask)
                    dice = dice_score(pmask, true_mask)
                    if iou > best_iou:
                        best_iou, best_dice = iou, dice
                ious.append(best_iou)
                dices.append(best_dice)
                per_img_ious.append(best_iou)
                per_img_dices.append(best_dice)

            image_rows.append({
                "batch_idx": batch_idx,
                "num_ground_truth": len(true_masks),
                "num_predictions": len(pred_masks),
                "mean_iou": float(np.mean(per_img_ious)) if per_img_ious else 0.0,
                "mean_dice": float(np.mean(per_img_dices)) if per_img_dices else 0.0,
            })

    results = {
        "score_threshold": score_threshold,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "avg_inference_time": float(np.mean(inference_times)) if inference_times else 0.0,
    }

    print("Evaluation Results")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()) if image_rows else [])
            if image_rows:
                writer.writeheader()
                writer.writerows(image_rows)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"])

    image_key = f"{args.split}_images"
    ann_key = f"{args.split}_annotations"
    ds = PlantCocoDataset(cfg["data"][image_key], cfg["data"][ann_key], cfg["training"]["image_size"])
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=cfg["training"].get("num_workers", 0),
                                         collate_fn=collate_fn)

    model = get_maskrcnn_model(cfg["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    csv_path = f"outputs/metrics/eval_{args.split}_thr_{args.threshold}.csv"
    evaluate(model, loader, device, args.threshold, csv_path)


if __name__ == "__main__":
    main()
