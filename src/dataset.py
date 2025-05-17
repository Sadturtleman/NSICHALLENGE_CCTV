from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch
from torchvision import transforms

class CrowdDataset(Dataset):
    def __init__(self, image_dir, density_dir, transform=None):
        self.image_dir = image_dir
        self.density_dir = density_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        density_path = os.path.join(self.density_dir, image_name.replace('.jpg', '.npy'))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        density_map = np.load(density_path)
        original_sum = density_map.sum()
        density_map = cv2.resize(density_map, (64, 64), interpolation=cv2.INTER_LINEAR)

        if density_map.sum() > 0:
            density_map *= (original_sum / density_map.sum())

        density_map = torch.from_numpy(density_map).unsqueeze(0)

        return image, density_map
