import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
from source.data_loader import MNISTWithBBox, LatentDataset, AugmentedDataset
from source.vae import VAEWithBoundingBox, train_vae_with_bbox, save_model, save_vae_outputs
from source.gan import Generator, Discriminator, train_gan
from source.ocr import CRNN, train_crnn, evaluate_crnn, generate_and_save_labels
from source.utils import load_config

def load_ocr_model(model_path, imgH, nc, nclass, nh, device):
    model = CRNN(imgH=imgH, nc=nc, nclass=nclass, nh=nh)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def image_to_text(image_path, model, transform, device):
    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        output = output.permute(1, 0, 2)  # [w, b, c]
        _, predicted = torch.max(output, 2)
        predicted = predicted.transpose(1, 0).contiguous().view(-1)
        predicted_text = "".join([str(p.item()) for p in predicted if p.item() != 0])
    return predicted_text

def main(config):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VAE data
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    mnist_with_bbox = MNISTWithBBox(train=True, transform=transform)
    vae_dataloader = DataLoader(mnist_with_bbox, batch_size=128, shuffle=True)

    # Train VAE
    latent_dim = config['vae']['latent_dim']
    vae_bbox = VAEWithBoundingBox(latent_dim).to(device)
    train_vae_with_bbox(vae_bbox, vae_dataloader, device, epochs=config['vae']['epochs'], lr=config['vae']['lr'])

    # Save VAE outputs
    vae_output_dir = config['vae']['output_dir']
    os.makedirs(vae_output_dir, exist_ok=True)
    save_model(vae_bbox, config['vae']['model_save_path'])
    save_vae_outputs(vae_bbox, vae_dataloader, vae_output_dir, device)

    # Load GAN data
    latent_dataset = LatentDataset(config['vae']['output_dir'], transform=transform)
    gan_dataloader = DataLoader(latent_dataset, batch_size=config['gan']['batch_size'], shuffle=True)

    # Train GAN and save synthetic data
    img_shape = (1, 32, 32)
    generator = Generator(latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)
    train_gan(generator, discriminator, gan_dataloader, latent_dim, img_shape, device,
              config['gan']['model_dir'], config['gan']['synthetic_dir'], config['gan']['augmented_dir'], 
              epochs=config['gan']['epochs'], lr=config['gan']['lr'])

    # Generate synthetic data labels
    generate_and_save_labels(config['ocr']['data_dir'])

    # Load OCR data
    augment_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(1, 32, 32))
    ])
    augmented_dataset = AugmentedDataset(config['ocr']['data_dir'], transform=augment_transform)
    ocr_dataloader = DataLoader(augmented_dataset, batch_size=config['ocr']['batch_size'], shuffle=True)

    # Train CRNN model
    crnn_model = CRNN(imgH=32, nc=1, nclass=config['ocr']['num_classes'], nh=256).to(device)
    train_crnn(crnn_model, ocr_dataloader, device, epochs=config['ocr']['epochs'], lr=config['ocr']['lr'])

    # Evaluate CRNN model
    evaluate_crnn(crnn_model, ocr_dataloader, device)

    # Save CRNN model
    torch.save(crnn_model.state_dict(), config['ocr']['model_save_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OCR Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--test', type=str, help='Path to an image file to test the OCR model')
    args = parser.parse_args()
    config = load_config(args.config)

    if args.test:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        crnn_model = load_ocr_model(config['ocr']['model_save_path'], imgH=32, nc=1, nclass=config['ocr']['num_classes'], nh=256, device=device)
        test_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        predicted_text = image_to_text(args.test, crnn_model, test_transform, device)
        print(f"Predicted text: {predicted_text}")
    else:
        main(config)
