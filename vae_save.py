import torch
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from VAE import VAEWithBoundingBox, MNISTWithBBox, train_vae_with_bbox

# Define file paths for saving the model and outputs
MODEL_SAVE_PATH = './vae_bbox_model.pth'
OUTPUT_SAVE_DIR = './vae_outputs/'

# Ensure the output directory exists
os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)

# Function to save model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# Function to generate and save VAE outputs
def save_vae_outputs(model, dataloader, output_dir, num_batches=10):
    model.eval()
    idx = 0
    with torch.no_grad():
        for batch_idx, (data, bbox) in enumerate(dataloader):
            data = data.view(-1, 784)
            recon_batch, pred_bbox, mu, logvar = model(data)
            
            for i in range(data.size(0)):
                original_img = data[i].view(1, 28, 28).cpu().numpy()
                reconstructed_img = recon_batch[i].view(1, 28, 28).cpu().numpy()
                bbox = pred_bbox[i].cpu().numpy()
                
                np.save(os.path.join(output_dir, f'original_img_{idx}.npy'), original_img)
                np.save(os.path.join(output_dir, f'reconstructed_img_{idx}.npy'), reconstructed_img)
                np.save(os.path.join(output_dir, f'bbox_{idx}.npy'), bbox)
                idx += 1

            if batch_idx >= num_batches - 1:
                break
                
    print(f'VAE outputs saved to {output_dir}')

# Load the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_with_bbox = MNISTWithBBox(train=True, transform=transform)
dataloader = DataLoader(mnist_with_bbox, batch_size=128, shuffle=True)

# Initialize the VAE model
latent_dim = 20
vae_bbox = VAEWithBoundingBox(latent_dim)

# Check if the model save path exists
if os.path.exists(MODEL_SAVE_PATH):
    # Load the saved model
    vae_bbox.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print(f'Model loaded from {MODEL_SAVE_PATH}')
else:
    # Train the model if it hasn't been saved yet
    print('Training the model...')
    train_vae_with_bbox(vae_bbox, dataloader)
    # Save the trained model
    save_model(vae_bbox, MODEL_SAVE_PATH)

# Generate and save the VAE outputs
save_vae_outputs(vae_bbox, dataloader, OUTPUT_SAVE_DIR)
