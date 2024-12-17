# Databricks notebook source
import optuna
import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

# ---- Data Loading ----
BASE_DIR = pathlib.Path('/Volumes/pmr_dev/lead_ingots/lead_ingot_images')
image_dir = BASE_DIR / 'Ingot'
label_dir = BASE_DIR / 'Labels/Labels.csv'
file_list = pd.read_csv(label_dir)

# Encode labels
def encode_labels(file_list):
    image_names = file_list['Image']
    labels = file_list.drop(columns=['Image']).apply(
        lambda row: '_'.join(row.index[row == True]), axis=1
    )
    return dict(zip(image_names, labels))

label_dict = encode_labels(file_list)
image_names = file_list['Image'].tolist()

# Custom Dataset
class LeadIngotDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load images
def load_images(image_names, image_dir):
    images = []
    for name in image_names:
        path = image_dir / name
        image = Image.open(path).convert('RGB')
        images.append(image)
    return images

# ---- Autoencoder Model ----
class Autoencoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels[0], 3, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ---- Optuna Objective Function ----
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    channels = [
        trial.suggest_int('ch1', 16, 64),
        trial.suggest_int('ch2', 32, 128),
        trial.suggest_int('ch3', 64, 256),
    ]

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Filter label 7 only
    filtered_indices = [i for i, name in enumerate(image_names) if label_dict[name] == 'goede_blok']
    filtered_images = load_images([image_names[i] for i in filtered_indices], image_dir)
    filtered_labels = [0] * len(filtered_indices)

    dataset = LeadIngotDataset(filtered_images, filtered_labels, transform)
    train_val_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.125, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

    # Model, loss, and optimizer
    model = Autoencoder(channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training Loop
    patience, best_loss, patience_counter = 5, float('inf'), 0
    for epoch in range(20):  # Max epochs
        model.train()
        train_loss = 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            loss = loss_fn(outputs, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                outputs = model(x_batch)
                val_loss += loss_fn(outputs, x_batch).item()

        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    return best_loss

# ---- Optuna Study ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# ---- Best Parameters ----
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

