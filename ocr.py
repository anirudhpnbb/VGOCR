import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os

# Generate Labels if Not Exist
def generate_and_save_labels(data_dir):
    image_files = sorted([f for f in os.listdir(data_dir) if 'img' in f])
    label_files = sorted([f for f in os.listdir(data_dir) if 'label' in f])
    
    if len(label_files) == 0:
        print("No label files found. Generating labels...")
        num_images = len(image_files)
        labels = np.random.randint(0, 10, size=num_images)  # Example: Generate random labels for 10 classes
        
        for idx, image_file in enumerate(image_files):
            label = labels[idx]
            label_filename = image_file.replace('img', 'label')
            label_path = os.path.join(data_dir, label_filename)
            np.save(label_path, label)
            print(f'Saved label {label} to {label_path}')
    else:
        print("Label files found. Skipping label generation.")

data_dir = '/home/anirudh/Desktop/Projects/Personal/VGOCR/augmented_data'
generate_and_save_labels(data_dir)

# Define Dataset Class
class AugmentedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if 'img' in f])
        self.labels = sorted([f for f in os.listdir(data_dir) if 'label' in f])
        
        # Debugging statements
        print(f"Image files ({len(self.image_files)}): {self.image_files[:5]} ...")
        print(f"Label files ({len(self.labels)}): {self.labels[:5]} ...")

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
        
        return image, torch.tensor(label, dtype=torch.long)

# Define Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    transforms.Lambda(lambda x: x.view(1, 28, 28))  # Ensure correct shape
])

# Create Dataset and DataLoader
dataset = AugmentedDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check Dataset Length
print(f"Dataset Length: {len(dataset)}")

# Define the OCR Model
class OCRModel(nn.Module):
    def __init__(self, num_classes=10):  # Assuming 10 classes for simplicity
        super(OCRModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 128 * 3 * 3)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10  # Adjust this based on your dataset
model = OCRModel(num_classes).to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 batches
            print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print('Finished Training')

# Evaluate the Model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total} %')

# Save the Model
model_save_path = '/home/anirudh/Desktop/Projects/Personal/VGOCR/ocr_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
