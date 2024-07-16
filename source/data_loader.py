import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

# Helper function to calculate bounding box
def get_bounding_box(img):
    img = img.squeeze()
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax, rmax]

class MNISTWithBBox(Dataset):
    def __init__(self, train=True, transform=None):
        self.dataset = datasets.MNIST('./data', train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        bbox = get_bounding_box(np.array(img))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(bbox, dtype=torch.float32)

class LatentDataset(Dataset):
    def __init__(self, latent_dir, transform=None):
        self.latent_dir = latent_dir
        self.transform = transform
        self.latent_files = sorted([f for f in os.listdir(latent_dir) if 'latent_vector' in f])
        self.image_files = sorted([f for f in os.listdir(latent_dir) if 'reconstructed_img' in f])
        assert len(self.latent_files) == len(self.image_files), "Mismatch in number of latent vectors and images"

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        image_path = os.path.join(self.latent_dir, self.image_files[idx])
        latent_vector = np.load(latent_path)
        real_image = np.load(image_path)
        real_image = real_image.squeeze()  # Ensure the image is 2D
        real_image = Image.fromarray((real_image * 255).astype(np.uint8))  # Convert numpy array to PIL Image
        if self.transform:
            real_image = self.transform(real_image)
        return torch.tensor(latent_vector, dtype=torch.float32), real_image

class AugmentedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if 'img' in f])
        self.labels = sorted([f for f in os.listdir(data_dir) if 'label' in f])
        assert len(self.image_files) == len(self.labels), "Mismatch in number of images and labels"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        label_path = os.path.join(self.data_dir, self.labels[idx])
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file {label_path} does not exist.")
        
        image = np.load(image_path)
        image = image.squeeze()  # Ensure the image is 2D
        image = Image.fromarray((image * 255).astype(np.uint8))  # Convert numpy array to PIL Image
        
        if self.transform:
            image = self.transform(image)
        
        label = np.load(label_path)
        if label.ndim == 0:  # Handle case where label is a scalar
            label = np.expand_dims(label, axis=0)
        label = torch.from_numpy(label).long()  # Ensure label is a long tensor
        
        return image, label, len(label)
