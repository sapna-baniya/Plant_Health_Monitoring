import os
import torch
from PIL import Image
import torchvision.transforms.functional as F

class SemanticPlantDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        image_path = os.path.join(self.image_dir, filename)
        mask_name = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image = F.to_tensor(image)
        mask = torch.as_tensor(list(mask.getdata()), dtype=torch.long)
        mask = mask.reshape(self.image_size, self.image_size)

        return image, mask
