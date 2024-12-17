# Databricks notebook source
import pathlib

BASE_DIR= pathlib.Path('/Volumes/pmr_adp_lrin_dev/playground/neu_steel')

train_dir= BASE_DIR / 'NEU-DET'
file_list= list(train_dir.glob('**/**/*.jpg'))



# COMMAND ----------

print(file_list[0])

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

def encode_labels(file_list):
    label_dict = {}
    for file in file_list:
        file_name = file.name
        label = file_name.split('_')[0]
        label_dict[file_name] = label

    le = LabelEncoder()
    encoded_labels = le.fit_transform(list(label_dict.values()))
    label_dict = dict(zip(label_dict.keys(), encoded_labels))
    
    return label_dict

label_dict= encode_labels(file_list)


# COMMAND ----------

from PIL import Image

voorbeeld= Image.open(file_list[0])
voorbeeld.show()
voorbeeld.size

# COMMAND ----------

import torch
from torch.utils.data import Dataset
# Custom Dataset Class
class LeadIngotDataset(Dataset):
    def __init__(self, image_paths, images, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.images = images  
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# COMMAND ----------

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)), #alexnet, vgg16 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet noramlization
])

# COMMAND ----------

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

batch_size= 32

dataset= SteelDefectDataset(file_list, label_dict, transform)

train_val_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=0.2, stratify=list(label_dict.values()), random_state=42
)

# Then split train+val into train and validation sets
train_indices, val_indices = train_test_split(
    train_val_indices, test_size=0.125, stratify=[label_dict[dataset.image_files[i].name] for i in train_val_indices], random_state=42
)

# Create Subset datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Print statistics
train_labels = [label_dict[dataset.image_files[i].name] for i in train_indices]
val_labels = [label_dict[dataset.image_files[i].name] for i in val_indices]
test_labels = [label_dict[dataset.image_files[i].name] for i in test_indices]

print(f"Train set size: {len(train_dataset)} frequenties: {torch.bincount(torch.tensor(train_labels))}")
print(f"Validation set size: {len(val_dataset)} frequenties: {torch.bincount(torch.tensor(val_labels))}")
print(f"Test set size: {len(test_dataset)} frequenties: {torch.bincount(torch.tensor(test_labels))}")

# COMMAND ----------

import torch.nn as nn
from torch import nn
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()        
        # N, 1, 224, 224
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), # -> N, 16, 112, 112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> N, 32, 56, 56
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> N, 64, 28, 28
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> N, 128, 14, 14
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # -> N, 256, 7, 7
        )
        
        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # -> N, 128, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # -> N, 64, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # -> N, 32, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # -> N, 16, 112, 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), # -> N, 1, 224, 224
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# COMMAND ----------

model=Autoencoder()

#device = torch.device("cuda:0")
device = torch.device("cpu")

model = model.to(device)

# COMMAND ----------

#x= torch.ones((64,3,224,224))
#model(x).shape

# COMMAND ----------

import time
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, train_dl, valid_dl, patience):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    patience_counter = 0  # Initialize patience counter
    best_model_wts = model.state_dict()  # Initialize best model weights

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training Phase
        model.train()
        batch_train_count = 0
        for x_batch, y_batch in train_dl:
            batch_train_count += 1
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Track metrics
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

            if batch_train_count % 10 == 0:  # Print progress every 10 batches
                print(f"  [Training] Batch {batch_train_count}/{len(train_dl)} - Loss: {loss.item():.4f}")

        # Calculate epoch-level metrics for training
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        # Validation Phase
        model.eval()
        batch_valid_count = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                batch_valid_count += 1
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

                # Track metrics
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

                if batch_valid_count % 10 == 0:  # Print progress every 10 batches
                    print(f"  [Validation] Batch {batch_valid_count}/{len(valid_dl)} - Loss: {loss.item():.4f}")

        # Calculate epoch-level metrics for validation
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        end_time = time.time()
        epoch_duration = end_time - start_time

        # Early Stopping
        if epoch > 0 and loss_hist_valid[epoch] > min(loss_hist_valid[:epoch]):
            patience_counter += 1
        else:
            patience_counter = 0
            best_model_wts = model.state_dict()  # Save the best model weights

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            model.load_state_dict(best_model_wts)  # Load the best model weights
            break

        # Epoch Summary
        print(
            f"Epoch {epoch + 1}/{num_epochs} Summary:"
            f"\n  Training - Loss: {loss_hist_train[epoch]:.4f}, Accuracy: {accuracy_hist_train[epoch]:.4f}"
            f"\n  Validation - Loss: {loss_hist_valid[epoch]:.4f}, Accuracy: {accuracy_hist_valid[epoch]:.4f}"
            f"\n  Duration: {epoch_duration:.2f}s"
        )

    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

# Training parameters
num_epochs = 300
hist = train(model, num_epochs, train_loader, val_loader, patience=5)


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
early_stop_epoch = next((i for i, v in enumerate(hist[0]) if v == 0), len(hist[0]))
x_arr = np.arange(len(hist[0])) + 1

# Adjust x_arr to stop at early_stop_epoch
x_arr = x_arr[:early_stop_epoch]

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0][:early_stop_epoch], '-o', label='Train loss')
ax.plot(x_arr, hist[1][:early_stop_epoch], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2][:early_stop_epoch], '-o', label='Train acc.')
ax.plot(x_arr, hist[3][:early_stop_epoch], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Function to display images
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean
def imshow(img, ax):
    img = denormalize(img)
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))

# Get a batch of images from the no_defect_val_loader
dataiter = iter(no_defect_val_loader)
images, _ = next(dataiter)

# Move the images to the device
images = images.to(device)

# Get the reconstructed images
model.eval()
with torch.no_grad():
    reconstructed = model(images)

# Move the images back to CPU for plotting
images = images.cpu()
reconstructed = reconstructed.cpu()

# Plot the original and reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(12, 4))

for i in range(6):
    # Original images
    ax = axes[0, i]
    imshow(images[i], ax)
    ax.axis('off')
    if i == 0:
        ax.set_title('Original')

    # Reconstructed images
    ax = axes[1, i]
    imshow(reconstructed[i], ax)
    ax.axis('off')
    if i == 0:
        ax.set_title('Reconstructed')

plt.show()
