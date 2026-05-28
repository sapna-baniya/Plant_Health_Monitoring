import os
import random
import torch
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO
import numpy as np
import torchvision.transforms.functional as F


class PlantCocoDataset(torch.utils.data.Dataset):
    """COCO instance segmentation dataset for 3 plant classes.

    Expected category IDs:
      1 healthy_leaf
      2 healthy_stem
      3 diseased_leaf
    """
    def __init__(self, image_dir, annotation_path, image_size=512, transforms=None,
                 use_augmentation=False, aug_cfg=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.image_size = image_size
        self.transforms = transforms
        self.use_augmentation = use_augmentation
        self.aug_cfg = aug_cfg or {}

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        path = os.path.join(self.image_dir, img_info["file_name"])

        image = Image.open(path).convert("RGB")
        original_w, original_h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        width, height = image.size

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue

            # Create mask from segmentation if available, otherwise from bbox
            seg = ann.get("segmentation", [])

            use_segmentation = (
               isinstance(seg, list)
               and len(seg) > 0
               and isinstance(seg[0], list)
               and len(seg[0]) >= 6
            )

            if use_segmentation:
                try:
                   mask = self.coco.annToMask(ann)
                except Exception:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(width, int(x + w))
                    y2 = min(height, int(y + h))
                    mask[y1:y2, x1:x2] = 1
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(width, int(x + w))
                y2 = min(height, int(y + h))
                mask[y1:y2, x1:x2] = 1
            mask = Image.fromarray(mask.astype(np.uint8))

            mask_resized = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

            sx = self.image_size / original_w
            sy = self.image_size / original_h

            boxes.append([x * sx, y * sy, (x + w) * sx, (y + h) * sy])
            labels.append(ann["category_id"])
            masks.append(np.array(mask_resized, dtype=np.uint8))
            areas.append(float(ann.get("area", w * h)) * sx * sy)
            iscrowd.append(ann.get("iscrowd", 0))

        image = image.resize((self.image_size, self.image_size))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.use_augmentation:
            image, boxes, masks = self.apply_augmentation(image, boxes, masks)

        image = F.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }
        return image, target

    def apply_augmentation(self, image, boxes, masks):
        # Horizontal flip with correct box and mask transforms.
        flip_p = float(self.aug_cfg.get("horizontal_flip_prob", 0.5))
        if random.random() < flip_p and len(boxes) > 0:
            image = F.hflip(image)
            masks = torch.flip(masks, dims=[2])
            w = self.image_size
            old_x1 = boxes[:, 0].clone()
            old_x2 = boxes[:, 2].clone()
            boxes[:, 0] = w - old_x2
            boxes[:, 2] = w - old_x1

        # Photometric augmentation does not alter masks/boxes.
        brightness = float(self.aug_cfg.get("brightness", 0.0))
        contrast = float(self.aug_cfg.get("contrast", 0.0))
        saturation = float(self.aug_cfg.get("saturation", 0.0))
        hue = float(self.aug_cfg.get("hue", 0.0))

        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if contrast > 0:
            factor = 1.0 + random.uniform(-contrast, contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if saturation > 0:
            factor = 1.0 + random.uniform(-saturation, saturation)
            image = ImageEnhance.Color(image).enhance(factor)
        if hue > 0:
            # Simple HSV hue jitter using PIL/numpy.
            hsv = np.array(image.convert("HSV"), dtype=np.uint8)
            shift = int(random.uniform(-hue, hue) * 255)
            hsv[..., 0] = (hsv[..., 0].astype(int) + shift) % 255
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")

        return image, boxes, masks

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))
