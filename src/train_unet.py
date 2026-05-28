import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from models.mini_unet import MiniUNet
from dataset_semantic import SemanticPlantDataset
from utils.config import load_config
from utils.metrics import pixel_accuracy

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for images, masks in tqdm(loader, desc="Train" if train else "Val"):
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_acc += pixel_accuracy(logits, masks)

    return total_loss / max(len(loader), 1), total_acc / max(len(loader), 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"])

    train_mask_dir = "data/semantic_masks/train"
    val_mask_dir = "data/semantic_masks/val"

    train_ds = SemanticPlantDataset(cfg["data"]["train_images"], train_mask_dir, cfg["training"]["image_size"])
    val_ds = SemanticPlantDataset(cfg["data"]["val_images"], val_mask_dir, cfg["training"]["image_size"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["training"]["num_workers"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["training"]["num_workers"]
    )

    model = MiniUNet(num_classes=cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
    best_val = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            path = os.path.join(cfg["training"]["output_dir"], "mini_unet_best.pth")
            torch.save(model.state_dict(), path)
            print(f"Saved best model: {path}")

if __name__ == "__main__":
    main()
