import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# Hyperparameters
lr = 0.0002
batch_size = 64
image_size = 28  # MNIST images are 28x28
image_channels = 1  # Grayscale images
latent_dim = 20  # Example latent dimension from VAE
epochs = 100
save_interval = 10  # Save models and outputs every 10 epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories
latent_dir = '/home/anirudh/Desktop/Projects/Personal/VGOCR/vae_outputs'
gan_model_dir = '/home/anirudh/Desktop/Projects/Personal/VGOCR/gan_model'
synthetic_dir = '/home/anirudh/Desktop/Projects/Personal/VGOCR/synthetic_images'
augmented_dir = '/home/anirudh/Desktop/Projects/Personal/VGOCR/augmented_data'

# Ensure directories exist
os.makedirs(gan_model_dir, exist_ok=True)
os.makedirs(synthetic_dir, exist_ok=True)
os.makedirs(augmented_dir, exist_ok=True)

# Custom Dataset for Latent Vectors and Images
class LatentDataset(Dataset):
    def __init__(self, latent_dir, transform=None):
        self.latent_dir = latent_dir
        self.transform = transform
        self.latent_files = sorted([f for f in os.listdir(latent_dir) if 'latent_vector' in f])
        self.image_files = sorted([f for f in os.listdir(latent_dir) if 'reconstructed_img' in f])

        print("Latent Files:", self.latent_files)
        print("Image Files:", self.image_files)

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

# Define Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Create Dataset and DataLoader
latent_dataset = LatentDataset(latent_dir, transform=transform)
dataloader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=True)

# Check Dataset Length
print(f"Dataset Length: {len(latent_dataset)}")

# Build the GAN
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

img_shape = (image_channels, image_size, image_size)
generator = Generator(latent_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)

# Loss Function and Optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
for epoch in range(epochs):
    for i, (latent_vectors, real_images) in enumerate(dataloader):
        batch_size = latent_vectors.size(0)
        
        # Adversarial ground truths
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        # Configure input
        real_images = real_images.to(device)
        latent_vectors = latent_vectors.to(device)
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # Generate a batch of images
        gen_images = generator(latent_vectors)
        
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_images), valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_images), valid)
        fake_loss = adversarial_loss(discriminator(gen_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        if i % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} '
                   f'Loss D: {d_loss.item()}, loss G: {g_loss.item()}')

    # Save model and generated images at intervals
    if epoch % save_interval == 0:
        # Save the generator and discriminator models
        torch.save(generator.state_dict(), os.path.join(gan_model_dir, f'generator_epoch_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(gan_model_dir, f'discriminator_epoch_{epoch}.pth'))
        print(f'Saved model at epoch {epoch}')

        # Generate synthetic images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        fake_images = fake_images.cpu().detach().numpy()

        # Save synthetic images
        for idx, img in enumerate(fake_images):
            np.save(os.path.join(synthetic_dir, f'synthetic_img_epoch_{epoch}_{idx}.npy'), img)

        # Augment the dataset with synthetic images
        for idx, (real_img, fake_img) in enumerate(zip(real_images.cpu().numpy(), fake_images)):
            np.save(os.path.join(augmented_dir, f'augmented_real_img_{epoch}_{idx}.npy'), real_img)
            np.save(os.path.join(augmented_dir, f'augmented_synthetic_img_{epoch}_{idx}.npy'), fake_img)
        print(f'Saved synthetic images and augmented dataset at epoch {epoch}')

# Generate and Save Final Images for Visualization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

z = torch.randn(batch_size, latent_dim).to(device)
fake_images = generator(z)
fake_images = denorm(fake_images)
fake_images = fake_images.cpu().detach().numpy()

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(fake_images[i].transpose(1, 2, 0).squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
