import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(1024, 400)  # Adjusted from 784 to 1024
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

class DecoderWithBoundingBox(nn.Module):
    def __init__(self, latent_dim):
        super(DecoderWithBoundingBox, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 1024)  # Adjusted from 784 to 1024
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
        return recon_x, bbox, mu, logvar, z

def bbox_loss(pred_bbox, true_bbox):
    return nn.functional.mse_loss(pred_bbox, true_bbox, reduction='sum')

def loss_function_with_bbox(recon_x, x, pred_bbox, true_bbox, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BBL = bbox_loss(pred_bbox, true_bbox)
    return BCE + KLD + BBL

def train_vae_with_bbox(model, dataloader, device, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, bbox) in enumerate(dataloader):
            data, bbox = data.to(device), bbox.to(device)
            data = data.view(-1, 1024)  # Adjusted from 784 to 1024
            optimizer.zero_grad()
            recon_batch, pred_bbox, mu, logvar, _ = model(data)
            loss = loss_function_with_bbox(recon_batch, data, pred_bbox, bbox, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {train_loss / len(dataloader.dataset)}')

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def save_vae_outputs(model, dataloader, output_dir, device, num_batches=10):
    model.to(device)
    model.eval()
    idx = 0
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (data, bbox) in enumerate(dataloader):
            data, bbox = data.to(device), bbox.to(device)
            data = data.view(-1, 1024)  # Adjusted from 784 to 1024
            recon_batch, pred_bbox, mu, logvar, z = model(data)
            for i in range(data.size(0)):
                original_img = data[i].view(1, 32, 32).cpu().numpy()  # Adjusted from (1, 28, 28) to (1, 32, 32)
                reconstructed_img = recon_batch[i].view(1, 32, 32).cpu().numpy()  # Adjusted from (1, 28, 28) to (1, 32, 32)
                latent_vector = z[i].cpu().numpy()
                np.save(os.path.join(output_dir, f'original_img_{idx}.npy'), original_img)
                np.save(os.path.join(output_dir, f'reconstructed_img_{idx}.npy'), reconstructed_img)
                np.save(os.path.join(output_dir, f'latent_vector_{idx}.npy'), latent_vector)
                idx += 1
            if batch_idx >= num_batches - 1:
                break
    print(f'VAE outputs saved to {output_dir}')
