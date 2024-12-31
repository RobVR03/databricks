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
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

# COMMAND ----------


# ---- Data Loading ----
BASE_DIR = pathlib.Path('/Volumes/pmr_dev/lead_ingots/lead_ingot_images')
image_dir = BASE_DIR / 'Ingot'
label_dir = BASE_DIR / 'Labels/Labels.csv'
file_list = pd.read_csv(label_dir)

def calculate_reconstruction_errors(loader, model):
    all_errors = []
    for images, _ in loader:
        images = images.to(device)
        with torch.no_grad():
            reconstructed = model(images)
        images = images.cpu()
        reconstructed = reconstructed.cpu()
        errors = F.mse_loss(reconstructed, images, reduction='none')
        errors = errors.mean(dim=[1, 2, 3])
        all_errors.extend(errors.numpy())
    return all_errors

def encode_labels(file_list):
    # Extract image names
    image_names = file_list['Image']
    
    # Combine fault categories into a single label
    # Create a label string by concatenating fault category names where the value is True
    labels = file_list.drop(columns=['Image']).apply(
        lambda row: '_'.join(row.index[row == True]), axis=1
    )
    
    # Encode the labels using LabelEncoder
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # Create a dictionary mapping image names to encoded labels
    label_dict = dict(zip(image_names, encoded_labels))

    label_mapping = dict(zip(le.transform(le.classes_),le.classes_ ))
    
    return label_dict, label_mapping

# Encode the labels
label_dict, label_mapping = encode_labels(file_list)

value_to_find ="Goede blok"
Goed_label = next((k for k, v in label_mapping.items() if v == value_to_find), None)

image_names = file_list['Image'].tolist()

labels = [label_dict[img] for img in image_names]

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
def define_autoencoder(trial):
    layers = []
    encoder_channels = []

    num_encoder_layers = trial.suggest_int("num_encoder_layers", 5, 8)
    in_channels = 3
    for i in range(num_encoder_layers):
        # Enforce out_channels to be 16 * 2^i
        out_channels = 16 * (2 ** i)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        # layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        encoder_channels.append(out_channels)
        in_channels = out_channels

    # Decoder (reverse the encoder structure)
    for i in reversed(range(num_encoder_layers-1)):
        out_channels = encoder_channels[i]
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        # layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels

    # Final reconstruction layer
    layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=3, stride=2, padding=1, output_padding=1))

    return nn.Sequential(*layers)

# ---- Optuna Objective Function ----
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((2048, 512)), #alexnet, vgg16 input size
        transforms.ToTensor()
    ])

    # Filter label 7 only
    filtered_indices = [i for i, label in enumerate(labels) if label == Goed_label]
    filtered_images = load_images([image_names[i] for i in filtered_indices], image_dir)
    filtered_labels = [0] * len(filtered_indices)

    dataset = LeadIngotDataset(filtered_images, filtered_labels, transform)
    train_val_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.125, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

    goede_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Filter images and labels to include only label 8
    faulty_indices = [i for i, label in enumerate(labels) if label != Goed_label]
    faulty_images = load_images([image_names[i] for i in faulty_indices], image_dir)
    faulty_labels = [labels[i] for i in faulty_indices]
    faulty_subset = LeadIngotDataset(faulty_images, faulty_labels, transform=transform)

    # Create a DataLoader for the no_faulty subset
    faulty_loader = DataLoader(faulty_subset, batch_size=6, shuffle=False)

    # Model, loss, and optimizer
    model = define_autoencoder(trial).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Learning Rate Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(100):
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

        # Scheduler Step
        scheduler.step(val_loss)

        # Calculate reconstruction errors for faulty and no_faulty images
        faulty_errors = calculate_reconstruction_errors(faulty_loader, model)
        goede_errors = calculate_reconstruction_errors(goede_loader, model)
        mean_fault = np.mean(faulty_errors)
        mean_good = np.mean(goede_errors)

        trial.report(mean_fault-mean_good, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mean_fault-mean_good





# COMMAND ----------

# ---- Optuna Study ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

# ---- Best Parameters ----
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
