import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

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

transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_with_bbox = MNISTWithBBox(train=True, transform=transform)
dataloader = DataLoader(mnist_with_bbox, batch_size=128, shuffle=True)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

class DecoderWithBoundingBox(nn.Module):
    def __init__(self, latent_dim):
        super(DecoderWithBoundingBox, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        self.fc5 = nn.Linear(400, 4)

    def forward(self, z):
        h3 = torch.relu(self.fc3(z))
        reconstructed_img = torch.sigmoid(self.fc4(h3))
        bbox = self.fc5(h3)
        return reconstructed_img, bbox

class VAEWithBoundingBox(nn.Module):
    def __init__(self, latent_dim):
        super(VAEWithBoundingBox, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = DecoderWithBoundingBox(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x, bbox = self.decoder(z)
        return recon_x, bbox, mu, logvar, z  # Added z to the return values

def bbox_loss(pred_bbox, true_bbox):
    return nn.functional.mse_loss(pred_bbox, true_bbox, reduction='sum')

def loss_function_with_bbox(recon_x, x, pred_bbox, true_bbox, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BBL = bbox_loss(pred_bbox, true_bbox)
    return BCE + KLD + BBL

def train_vae_with_bbox(model, dataloader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, bbox) in enumerate(dataloader):
            data = data.view(-1, 784)
            optimizer.zero_grad()
            recon_batch, pred_bbox, mu, logvar, _ = model(data)  # Updated to get z
            loss = loss_function_with_bbox(recon_batch, data, pred_bbox, bbox, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {train_loss / len(dataloader.dataset)}')

latent_dim = 20
vae_bbox = VAEWithBoundingBox(latent_dim)
train_vae_with_bbox(vae_bbox, dataloader)

def visualize_output(model, dataloader, num_images=10):
    model.eval()
    with torch.no_grad():
        data_iter = iter(dataloader)
        images, _ = next(data_iter)
        images = images.view(-1, 784)
        
        recon_images, pred_bboxes, _, _, _ = model(images)  # Updated to get z
        
        images = images.view(-1, 1, 28, 28).cpu().numpy()
        recon_images = recon_images.view(-1, 1, 28, 28).cpu().numpy()
        pred_bboxes = pred_bboxes.cpu().numpy()

        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
        for i in range(num_images):
            # Original image
            ax = axes[i, 0]
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title('Original Image')
            ax.axis('off')
            
            # Reconstructed image with bounding box
            ax = axes[i, 1]
            ax.imshow(recon_images[i].squeeze(), cmap='gray')
            x1, y1, x2, y2 = pred_bboxes[i]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
            ax.add_patch(rect)
            ax.set_title('Reconstructed Image with BBox')
            ax.axis('off')
        
        plt.show()

# Visualize the output
visualize_output(vae_bbox, dataloader)
