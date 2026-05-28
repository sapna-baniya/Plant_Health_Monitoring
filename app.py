import gradio as gr
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import numpy as np

from src.models.maskrcnn_model import get_maskrcnn_model

CLASS_NAMES = [
    "__background__",
    "healthy_leaf",
    "healthy_stem",
    "diseased_leaf"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "outputs/checkpoints/maskrcnn_best.pth"

model = get_maskrcnn_model(num_classes=4)
checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


def predict(image, threshold=0.5):
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image).to(DEVICE)

    with torch.no_grad():
        output = model([image_tensor])[0]

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()
    masks = output["masks"].cpu()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    detected = 0

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < threshold:
            continue

        detected += 1
        x1, y1, x2, y2 = box.numpy()
        class_name = CLASS_NAMES[int(label)]

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            f"{class_name}: {score:.2f}",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="red", alpha=0.7)
        )

        mask_np = mask[0].numpy() > 0.5

        colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 4))
        colored_mask[mask_np] = [0, 1, 0, 0.35]  # green transparent mask

        ax.imshow(colored_mask)

    ax.axis("off")
    ax.set_title(f"Detected Objects: {detected}")

    output_path = "outputs/predictions/gradio_output.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    return output_path


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Plant Image"),
        gr.Slider(0.05, 0.9, value=0.5, step=0.05, label="Confidence Threshold")
    ],
    outputs=gr.Image(type="filepath", label="Labeled Prediction Output"),
    title="Plant Health Detection using Mask R-CNN",
    description="Upload a plant image and the system will label healthy leaves, healthy stems, and diseased leaves."
)

demo.launch(share=True)