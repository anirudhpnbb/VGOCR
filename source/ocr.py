import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'Image height has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        self.cnn = nn.Sequential()
        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            self.cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                self.cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            self.cnn.add_module(f'relu{i}', nn.ReLU(True))
        
        convRelu(0)
        self.cnn.add_module('pooling0', nn.MaxPool2d(2, 2))  # 64x16x16
        convRelu(1)
        self.cnn.add_module('pooling1', nn.MaxPool2d(2, 2))  # 128x8x8
        convRelu(2, True)
        convRelu(3)
        self.cnn.add_module('pooling2', nn.MaxPool2d((2, 2), (2, 2), (0, 0)))  # 256x4x4
        convRelu(4, True)
        convRelu(5)
        self.cnn.add_module('pooling3', nn.MaxPool2d((2, 2), (2, 2), (0, 0)))  # 512x2x2
        convRelu(6, True)

        self.rnn = nn.LSTM(512, nh, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        assert h == 1, "the height of the conv must be 1"
        x = x.squeeze(2).permute(0, 2, 1)  # [b, w, c]
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out)
        return output

def train_crnn(model, dataloader, device, epochs=100, lr=0.001):
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels, label_lengths) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)  # [w, b, c]
            output_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs, labels, output_lengths, label_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
        scheduler.step()
    print('Finished Training')

def evaluate_crnn(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, label_lengths in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)  # [w, b, c]
            _, predicted = torch.max(outputs, 2)
            predicted = predicted.transpose(1, 0).contiguous().view(-1)
            predicted_text = "".join([str(p.item()) for p in predicted if p != 0])
            total += len(labels)
            correct += sum([1 for p, t in zip(predicted_text, labels) if p == t])
    print(f'Accuracy of the model on the test images: {100 * correct / total} %')

def image_to_text(image_path, model, transform, device):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    model.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 2)
        predicted = predicted.squeeze().detach().cpu().numpy()
        predicted_text = "".join([str(p) for p in predicted if p != 0])
    return predicted_text

def generate_and_save_labels(data_dir):
    """
    Generates and saves labels for images in the specified directory.

    Args:
    - data_dir (str): Directory containing images.
    """
    image_files = sorted([f for f in os.listdir(data_dir) if 'img' in f])
    label_files = sorted([f for f in os.listdir(data_dir) if 'label' in f])
    
    if len(label_files) == 0:
        print("No label files found. Generating labels...")
        num_images = len(image_files)
        labels = [np.random.randint(0, 10, size=(5,)) for _ in range(num_images)]  # Save labels as sequences of length 5
        
        for idx, image_file in enumerate(image_files):
            label = labels[idx]
            label_filename = image_file.replace('img', 'label')
            label_path = os.path.join(data_dir, label_filename)
            np.save(label_path, label)
            print(f'Saved label {label} to {label_path}')
    else:
        print("Label files found. Skipping label generation.")

# Example usage:
if __name__ == "__main__":
    # Define parameters
    imgH = 32
    nc = 1
    nclass = 11  # Assuming 10 digits + 1 for the blank label
    nh = 256
    batch_size = 32

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((imgH, imgH * 4)),  # assuming width is 4x height
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset and DataLoader
    # Assuming a custom dataset class that returns images, labels, and label lengths
    train_dataset = datasets.ImageFolder(root="train_data", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.ImageFolder(root="test_data", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, device, and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("*********************". device, "***************************")
    model = CRNN(imgH, nc, nclass, nh)

    # Training
    train_crnn(model, train_loader, device, epochs=50)

    # Evaluation
    evaluate_crnn(model, test_loader, device)
