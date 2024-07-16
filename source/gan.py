# source/gan.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

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

def train_gan(generator, discriminator, dataloader, latent_dim, img_shape, device, gan_model_dir, synthetic_dir, augmented_dir, epochs=1000, lr=0.0002, save_interval=10):
    # Ensure directories exist
    os.makedirs(gan_model_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)
    os.makedirs(augmented_dir, exist_ok=True)

    adversarial_loss = nn.BCELoss().to(device)
    generator.to(device)
    discriminator.to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (latent_vectors, real_images) in enumerate(dataloader):
            batch_size = latent_vectors.size(0)
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            real_images = real_images.to(device)
            latent_vectors = latent_vectors.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            gen_images = generator(latent_vectors)
            g_loss = adversarial_loss(discriminator(gen_images), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_images), valid)
            fake_loss = adversarial_loss(discriminator(gen_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 200 == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} Loss D: {d_loss.item()}, loss G: {g_loss.item()}')

        # Save synthetic images and augmented data at intervals
        if epoch % save_interval == 0:
            torch.save(generator.state_dict(), os.path.join(gan_model_dir, f'generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(gan_model_dir, f'discriminator_epoch_{epoch}.pth'))
            print(f'Saved model at epoch {epoch}')

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            fake_images = fake_images.cpu().detach().numpy()

            for idx, img in enumerate(fake_images):
                np.save(os.path.join(synthetic_dir, f'synthetic_img_epoch_{epoch}_{idx}.npy'), img)
            for idx, (real_img, fake_img) in enumerate(zip(real_images.cpu().numpy(), fake_images)):
                np.save(os.path.join(augmented_dir, f'augmented_real_img_{epoch}_{idx}.npy'), real_img)
                np.save(os.path.join(augmented_dir, f'augmented_synthetic_img_{epoch}_{idx}.npy'), fake_img)
            print(f'Saved synthetic images and augmented dataset at epoch {epoch}')
