import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

def generate_and_save_labels(data_dir):
    image_files = sorted([f for f in os.listdir(data_dir) if 'img' in f])
    label_files = sorted([f for f in os.listdir(data_dir) if 'label' in f])
    if len(label_files) == 0:
        print("No label files found. Generating labels...")
        num_images = len(image_files)
        labels = np.random.randint(0, 10, size=num_images)
        for idx, image_file in enumerate(image_files):
            label = labels[idx]
            label_filename = image_file.replace('img', 'label')
            label_path = os.path.join(data_dir, label_filename)
            np.save(label_path, label)
            print(f'Saved label {label} to {label_path}')
    else:
        print("Label files found. Skipping label generation.")

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'Image height has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x16
        convRelu(1)
        cnn.add_module('pooling{1}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x8
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{2}'.format(2), nn.MaxPool2d((2, 2), (2, 2), (0, 0)))  # 256x4x4
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{3}'.format(3), nn.MaxPool2d((2, 2), (2, 2), (0, 0)))  # 512x2x2
        convRelu(6, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, batch_first=True),
            nn.Linear(nh * 2, nclass)
        )

    def forward(self, x):
        # Convolution layers
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height of the conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)  # [b, w, c]

        # RNN layers
        rnn = self.rnn(conv)
        return rnn

def train_crnn(model, dataloader, device, epochs=100, lr=0.001):
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels, label_lengths) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)  # [w, b, c]
            loss = criterion(outputs, labels, outputs_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
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

def image_to_text(image_path, model, transform):
    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 2)
        predicted = predicted.squeeze().detach().cpu().numpy()
        predicted_text = "".join([str(p) for p in predicted if p != 0])
    return predicted_text
