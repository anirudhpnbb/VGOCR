import torch
from torchvision import transforms
from PIL import Image
import argparse
from ocr import CRNN, image_to_text
from utils import load_config

def load_model(model_path, num_classes, device):
    model = CRNN(imgH=28, nc=1, nclass=num_classes, nh=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main(config, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the CRNN model
    crnn_model = load_model(config['ocr']['model_save_path'], config['ocr']['num_classes'], device)
    
    # Define the transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Predict the text from the image
    predicted_text = image_to_text(image_path, crnn_model, transform)
    
    print(f"Predicted text: {predicted_text}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR Detection')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config, args.image)
