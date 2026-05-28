import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.ops import nms
from models.maskrcnn_model import get_maskrcnn_model

CLASS_NAMES = [
    "__background__",
    "healthy_leaf",
    "healthy_stem",
    "diseased_leaf"
]
COLORS = {1: (0, 255, 0), 2: (0, 0, 255), 3: (0, 255, 255)}


def run_model(model, image_tensor, flip=False):
    if flip:
        image_tensor = torch.flip(image_tensor, dims=[2])
    output = model([image_tensor])[0]
    scores = output["scores"].detach().cpu()
    labels = output["labels"].detach().cpu()
    boxes = output["boxes"].detach().cpu()
    masks = output["masks"].detach().cpu()

    if flip:
        w = image_tensor.shape[2]
        old_x1 = boxes[:, 0].clone()
        old_x2 = boxes[:, 2].clone()
        boxes[:, 0] = w - old_x2
        boxes[:, 2] = w - old_x1
        masks = torch.flip(masks, dims=[3])
    return scores, labels, boxes, masks


def predict(image_path, weights_path, output_path="outputs/predictions/prediction.png",
            num_classes=4, score_threshold=0.10, image_size=512,
            top_k=8, nms_threshold=0.35, tta=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = get_maskrcnn_model(num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size
    resized = image.resize((image_size, image_size))
    image_tensor = F.to_tensor(resized).to(device)

    with torch.no_grad():
        s1, l1, b1, m1 = run_model(model, image_tensor, flip=False)
        if tta:
            s2, l2, b2, m2 = run_model(model, image_tensor, flip=True)
            scores = torch.cat([s1, s2])
            labels = torch.cat([l1, l2])
            boxes = torch.cat([b1, b2])
            masks = torch.cat([m1, m2])
        else:
            scores, labels, boxes, masks = s1, l1, b1, m1

    print("Raw predictions:", len(scores))
    keep = scores >= score_threshold
    scores, labels, boxes, masks = scores[keep], labels[keep], boxes[keep], masks[keep]
    print("After threshold:", len(scores))

    # Class-wise NMS keeps overlapping predictions only when they are truly duplicate same-class boxes.
    final_indices = []
    if len(scores) > 0:
        for cls in labels.unique():
            idx = torch.where(labels == cls)[0]
            keep_cls = nms(boxes[idx], scores[idx], nms_threshold)
            final_indices.extend(idx[keep_cls].tolist())
        final_indices = torch.tensor(final_indices, dtype=torch.long)
        scores, labels, boxes, masks = scores[final_indices], labels[final_indices], boxes[final_indices], masks[final_indices]
    print("After class-wise NMS:", len(scores))

    if len(scores) > top_k:
        idx = torch.argsort(scores, descending=True)[:top_k]
        scores, labels, boxes, masks = scores[idx], labels[idx], boxes[idx], masks[idx]
    print("Final predictions shown:", len(scores))

    image_np = np.array(resized).copy()
    overlay = image_np.copy()

    for i in range(len(scores)):
        score = scores[i].item()
        label = labels[i].item()
        class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"class_{label}"
        color = COLORS.get(label, (255, 255, 255))

        mask = masks[i, 0].numpy() > 0.5
        colored = np.zeros_like(image_np, dtype=np.uint8)
        colored[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.35, 0)

        x1, y1, x2, y2 = boxes[i].numpy().astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name}: {score:.2f}"
        cv2.rectangle(overlay, (x1, max(y1 - 22, 0)), (min(x1 + 210, image_size), y1), color, -1)
        cv2.putText(overlay, text, (x1 + 3, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if len(scores) == 0:
        print("WARNING: No predictions shown. Try --threshold 0.03 or verify weights/classes.")

    overlay = cv2.resize(overlay, (original_w, original_h))
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved prediction to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="outputs/predictions/prediction.png")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--nms_threshold", type=float, default=0.35)
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip test-time augmentation.")
    args = parser.parse_args()

    predict(args.image, args.weights, args.output, score_threshold=args.threshold,
            image_size=args.image_size, top_k=args.top_k,
            nms_threshold=args.nms_threshold, tta=args.tta)
