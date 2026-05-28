# Plant Health Monitoring - 3 Class Final Project

Final task: instance segmentation of whole-plant images using 3 manually annotated classes:

1. `healthy_leaf`
2. `unhealthy_leaf`
3. `stem`

The project uses a COCO-pretrained Mask R-CNN model in PyTorch, replaces the output heads for 3 custom classes, fine-tunes the model, applies data augmentation, performs random-search hyperparameter tuning, and applies a recent post-processing improvement using class-wise NMS, top-k filtering, confidence threshold tuning, and optional test-time augmentation (TTA).

## Required Dataset Structure

Place your new 200 annotated samples here:

```text
data/
  images/
    train/
    val/
    test/
  annotations/
    train.json
    val.json
    test.json
```

Split must be 80/10/10:

```text
train: 160 images
val: 20 images
test: 20 images
```

COCO categories must be:

```json
[
  {"id": 1, "name": "healthy_leaf"},
  {"id": 2, "name": "unhealthy_leaf"},
  {"id": 3, "name": "stem"}
]
```

## If your COCO file still has old 8 classes

Run:

```bash
python src/remap_coco_to_3classes.py --input data/annotations/all_old.json --output data/annotations/all_3class.json
```

Then split:

```bash
python src/split_coco_dataset.py --images data/images/all --annotations data/annotations/all_3class.json --out data
```

## Validate dataset

```bash
python src/validate_coco.py --annotations data/annotations/train.json --image_dir data/images/train
python src/validate_coco.py --annotations data/annotations/val.json --image_dir data/images/val
python src/validate_coco.py --annotations data/annotations/test.json --image_dir data/images/test
```

## Baseline before fine-tuning

This evaluates the adapted pretrained network before plant-specific training.

```bash
python src/evaluate_baseline_no_training.py --config configs/maskrcnn_config.yaml --split val --threshold 0.05
```

## Train / fine-tune Mask R-CNN

```bash
python src/train_maskrcnn.py --config configs/maskrcnn_config.yaml
```

This uses:
- head-only training for first 3 epochs
- full backbone fine-tuning afterward
- augmentation
- training/validation loss logging
- best checkpoint saving

## Hyperparameter random search

```bash
python src/random_search_maskrcnn.py --config configs/maskrcnn_config.yaml --max_trials 4 --epochs 5
```

## Evaluate final model

```bash
python src/evaluate_maskrcnn.py --config configs/maskrcnn_config.yaml --weights outputs/checkpoints/maskrcnn_best.pth --split test --threshold 0.05
```

## Prediction visualization

```bash
python src/predict_maskrcnn.py \
--image data/images/test/YOUR_IMAGE.jpg \
--weights outputs/checkpoints/maskrcnn_best.pth \
--threshold 0.05 \
--top_k 6 \
--tta
```

## Full visualizations

```bash
python src/full_visualizations.py
```

Outputs go to:

```text
outputs/visualizations/
```

## Mini-network from scratch

The Mini U-Net baseline is included for Part 2. To run it, first convert COCO annotations into semantic masks:

```bash
python src/coco_to_semantic_masks.py --annotations data/annotations/train.json --output_dir data/semantic_masks/train
python src/coco_to_semantic_masks.py --annotations data/annotations/val.json --output_dir data/semantic_masks/val
python src/train_unet.py --config configs/unet_config.yaml
```

## Report wording for recent technique

Recent technique used: post-processing optimization for instance segmentation using confidence threshold tuning, class-wise Non-Max Suppression, top-k filtering, and optional test-time augmentation.

