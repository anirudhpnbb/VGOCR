import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import torch

# Helper function to calculate bounding box
def get_bounding_box(img):
    img = img.squeeze()  # Remove single-dimensional entries from the shape of an array
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
        image = np.load(image_path)
        label = np.load(label_path)
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(label)
        label_length = len(label)
        return image, label, label_length