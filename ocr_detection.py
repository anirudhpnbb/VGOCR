import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ocr import OCRModel  # Assuming the OCR model is defined in ocr_model.py

# Load the trained OCR model
model = OCRModel()
model.load_state_dict(torch.load('ocr_model.pth'))
model.eval()

# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

def image_to_text(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Apply the transformations
    image = transform(image)
    
    # Add a batch dimension
    image = image.unsqueeze(0)
    
    # Move the image tensor to the same device as the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)
    
    # Perform the forward pass
    with torch.no_grad():
        output = model(image)
    
    # Get the predicted label
    _, predicted = torch.max(output, 1)
    
    # Convert the predicted tensor to a numpy array and then to a list
    predicted_label = predicted.cpu().numpy().tolist()
    
    return predicted_label[0]

# Example usage
image_path = '/home/anirudh/Desktop/Projects/Personal/VGOCR/test/digit.jpg'
predicted_text = image_to_text(image_path)
print(f'Predicted Text: {predicted_text}')
