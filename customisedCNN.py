import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd  # For saving predictions to CSV
from PIL import Image
from torch.utils.data import Dataset

# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define paths
train_path = "C://Wildfire-Prediction-from-Satellite-Imagery-main//data//train"
valid_path = "C://Wildfire-Prediction-from-Satellite-Imagery-main//data//valid"
test_path = "C://Wildfire-Prediction-from-Satellite-Imagery-main//data//test"
wildfire = 'wildfire'
nowildfire = "nowildfire"

# Load test wildfire and no-wildfire files
test_wildfire_file_path = os.path.join(test_path, wildfire)
test_wildfire_files = [os.path.join(test_wildfire_file_path, file) for file in os.listdir(test_wildfire_file_path) if
                       file.endswith(".jpg")]
test_wildfire_files.sort()
test_nowildfire_file_path = os.path.join(test_path, nowildfire)
test_nowildfire_files = [os.path.join(test_nowildfire_file_path, file) for file in os.listdir(test_nowildfire_file_path)
                         if file.endswith(".jpg")]
test_nowildfire_files.sort()

# Load validation wildfire and no-wildfire files
valid_wildfire_file_path = os.path.join(valid_path, wildfire)
valid_wildfire_files = [os.path.join(valid_wildfire_file_path, file) for file in os.listdir(valid_wildfire_file_path) if
                        file.endswith(".jpg")]
valid_nowildfire_file_path = os.path.join(valid_path, nowildfire)
valid_nowildfire_files = [os.path.join(valid_nowildfire_file_path, file) for file in
                          os.listdir(valid_nowildfire_file_path) if file.endswith(".jpg")]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):  # Corrected __init__ method
        self.root_dir = root_dir
        self.transform = transform
        self.labels = os.listdir(self.root_dir)
        self.path_list = []
        self.label_list = []
        for label in self.labels:
            for filename in os.listdir(os.path.join(self.root_dir, label)):
                file_path = os.path.join(self.root_dir, label, filename)
                try:
                    Image.open(file_path).convert("RGB")
                    if os.path.isfile(file_path) and (
                            filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
                        self.path_list.append(file_path)
                        self.label_list.append(int(label == "wildfire"))
                except:
                    pass

    def __len__(self):  # Corrected __len__ method
        return len(self.label_list)

    def __getitem__(self, idx):  # Corrected __getitem__ method
        file_path = self.path_list[idx]
        label = self.label_list[idx]
        img = Image.open(file_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Set image transformation
image_size = 256
batch_size = 32
transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

# Load datasets
test_dataset = CustomDataset(root_dir=test_path, transform=transform)
valid_dataset = CustomDataset(root_dir=valid_path, transform=transform)

# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# Define the classification network
class ClassificationNetwork(nn.Module):
    def __init__(self, num_input_channels: int = 3):  # Corrected __init__ method
        super(ClassificationNetwork, self).__init__()
        input_channels = num_input_channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.avg_pool = nn.AvgPool2d(kernel_size=12, stride=1)
        self.linear = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        out = self.max_pool(self.batch_norm1(F.relu(self.conv1(x))))
        out = self.max_pool(self.batch_norm2(F.relu(self.conv2(out))))
        out = self.max_pool(self.batch_norm3(F.relu(self.conv3(out))))
        out = self.max_pool(self.batch_norm4(F.relu(self.conv4(out))))
        out = self.avg_pool(F.relu(self.conv5(out))).squeeze()
        out = F.softmax(self.linear(out), dim=1)
        return out

# Load models
model_folder = "C://Wildfire-Prediction-from-Satellite-Imagery-main//model"
num_of_model = 5
model_list = []
for i in range(num_of_model):
    test_model = ClassificationNetwork().to(device)
    test_model.load_state_dict(torch.load(os.path.join(model_folder, "training{}".format(i),
                                                       "best_test_model.pth"), map_location=device))
    valid_model = ClassificationNetwork().to(device)
    valid_model.load_state_dict(torch.load(os.path.join(model_folder, "training{}".format(i),
                                                        "best_valid_model.pth"), map_location=device))
    model_list.append([test_model, valid_model])

# Evaluate models and save predictions
num_of_test = 10
best_accuracy = 0
best_model_index = None
csv_data = []  # To store the predictions

for i in range(len(model_list)):
    for j in range(len(model_list[i])):
        model = model_list[i][j]
        for test_iter in range(num_of_test):
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
                    csv_data.extend([(label.item(), pred.item()) for label, pred in zip(labels, predicted)])

            accuracy = 100 * correct / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_index = (i, j)

            if j == 0:
                print("Accuracy of best test model of training{} in test data is {} %".format(i, accuracy))
            else:
                print("Accuracy of best train model of training{} in test data is {} %".format(i, accuracy))

# Save the CSV file with predictions
csv_df = pd.DataFrame(csv_data, columns=["Label", "Prediction"])
csv_df.to_csv("predictions.csv", index=False)

print("Best Accuracy: {} %".format(best_accuracy))
print("Finished Training and Saving Predictions to CSV")
