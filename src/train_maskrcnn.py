import argparse
import csv
import os
import torch
from tqdm import tqdm
from dataset_coco import PlantCocoDataset, collate_fn
from models.maskrcnn_model import get_maskrcnn_model
from utils.config import load_config


def set_backbone_trainable(model, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    loss_components = {}

    for images, targets in tqdm(loader, desc="Training"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v.item()

    n = max(len(loader), 1)
    loss_components = {k: v / n for k, v in loss_components.items()}
    return total_loss / n, loss_components


@torch.no_grad()
def validation_loss(model, loader, device):
    # Torchvision detection models return losses only in train mode.
    model.train()
    total_loss = 0.0
    loss_components = {}

    for images, targets in tqdm(loader, desc="Validation"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v.item()

    n = max(len(loader), 1)
    loss_components = {k: v / n for k, v in loss_components.items()}
    return total_loss / n, loss_components


def make_optimizer(model, cfg, backbone_trainable):
    lr = cfg["training"]["learning_rate"]
    backbone_lr = cfg["training"].get("backbone_learning_rate", lr * 0.1)
    wd = cfg["training"].get("weight_decay", 0.0001)
    momentum = cfg["training"].get("momentum", 0.9)

    if backbone_trainable:
        params = [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")], "lr": lr},
        ]
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)


def append_log(path, row, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"])
    out_dir = cfg["training"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_ds = PlantCocoDataset(
        cfg["data"]["train_images"],
        cfg["data"]["train_annotations"],
        cfg["training"]["image_size"],
        use_augmentation=cfg["training"].get("use_augmentation", True),
        aug_cfg=cfg.get("augmentation", {}),
    )
    val_ds = PlantCocoDataset(
        cfg["data"]["val_images"],
        cfg["data"]["val_annotations"],
        cfg["training"]["image_size"],
        use_augmentation=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate_fn,
    )

    model = get_maskrcnn_model(cfg["num_classes"]).to(device)

    freeze_epochs = int(cfg["training"].get("freeze_backbone_epochs", 0))
    set_backbone_trainable(model, False if freeze_epochs > 0 else True)
    optimizer = make_optimizer(model, cfg, backbone_trainable=(freeze_epochs == 0))

    best_val = float("inf")
    patience = int(cfg["training"].get("early_stopping_patience", 9999))
    no_improve = 0
    log_path = os.path.join(out_dir, "training_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)

    header = [
        "epoch", "phase", "train_loss", "val_loss",
        "train_loss_classifier", "train_loss_box_reg", "train_loss_mask", "train_loss_objectness", "train_loss_rpn_box_reg",
        "val_loss_classifier", "val_loss_box_reg", "val_loss_mask", "val_loss_objectness", "val_loss_rpn_box_reg",
    ]

    for epoch in range(cfg["training"]["epochs"]):
        if epoch == freeze_epochs and freeze_epochs > 0:
            print("Unfreezing backbone for full fine-tuning...")
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(model, cfg, backbone_trainable=True)

        phase = "head_only" if epoch < freeze_epochs else "full_finetune"
        train_loss, train_parts = train_one_epoch(model, optimizer, train_loader, device)
        val_loss, val_parts = validation_loss(model, val_loader, device)

        print(f"Epoch {epoch+1}: phase={phase}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        row = {"epoch": epoch + 1, "phase": phase, "train_loss": train_loss, "val_loss": val_loss}
        for k in ["loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"]:
            row[f"train_{k}"] = train_parts.get(k, 0.0)
            row[f"val_{k}"] = val_parts.get(k, 0.0)
        append_log(log_path, row, header)

        torch.save(model.state_dict(), os.path.join(out_dir, "maskrcnn_latest.pth"))
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            best_path = os.path.join(out_dir, "maskrcnn_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without validation improvement.")
                break


if __name__ == "__main__":
    main()
