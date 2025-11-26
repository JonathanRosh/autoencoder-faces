import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from model import Autoencoder
from datetime import datetime
import os
import time
import glob
import numpy as np
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.backends.cudnn.benchmark = True  # speed for fixed-size convs


class CustomImageDataset(data.Dataset):
    """
    Dataset over a list of file paths.
    Supports PNG/JPG or pre-saved .pt tensors.
    """
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        if img_path.endswith(".pt"):
            return torch.load(img_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_dataloaders(data_root, val_size, batch_size):
    """Create train/val dataloaders with 256x256 normalized images."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image_list = list(glob.glob(os.path.join(data_root, '*.[jp][pn]g'))) \
               + list(glob.glob(os.path.join(data_root, '*.pt')))
    if not image_list:
        raise FileNotFoundError(f"No images found in {data_root}. Check path and file types.")

    image_list.sort()
    val_paths = image_list[:val_size]
    train_paths = image_list[val_size:]
    
    print(f"Total paths found: {len(image_list)}")
    print(f"Training paths: {len(train_paths)}, Validation paths: {len(val_paths)}")

    train_dataset = CustomImageDataset(train_paths, transform)
    val_dataset = CustomImageDataset(val_paths, transform)

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, val_dataloader


def train_model(net, train_loader, val_loader, criterion, optimizer, num_epochs,
                model_name="unknown", log_dir=None):
    """
    Train for num_epochs and return a history dict with train/val loss per epoch.
    """
    net.train()
    start_time = time.time()

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        
        for i, real_images in enumerate(train_loader):
            real_images = real_images.to(DEVICE)
            
            optimizer.zero_grad()
            reconstructed_images = net(real_images)
            loss = criterion(reconstructed_images, real_images)
            loss.backward()
            optimizer.step()
            
            loss_value = loss.item()
            running_loss += loss_value
            epoch_loss_sum += loss_value
            epoch_batch_count += 1
            
            if i % 50 == 49:
                print(f'[{epoch+1}/{num_epochs}][{i+1}/{len(train_loader)}] Loss: {running_loss / 50:.6f}')
                running_loss = 0.0

        avg_train_loss = epoch_loss_sum / max(epoch_batch_count, 1)

        # validation loop
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for real_val_images in val_loader:
                real_val_images = real_val_images.to(DEVICE)
                reconstructed_val_images = net(real_val_images)
                val_loss += criterion(reconstructed_val_images, real_val_images).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1} finished in {time.time()-epoch_start_time:.2f}s. '
              f'Train Loss: {avg_train_loss:.6f}  Validation Loss: {avg_val_loss:.6f}')

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        net.train()

    print(f"Training finished. Total time: {(time.time() - start_time):.2f} seconds")

    # save history for plotting
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(log_dir, f"{model_name}_history_{timestamp}.pt")
        torch.save(history, log_path)
        print(f"Saved training history to {log_path}")

    return history


if __name__ == '__main__':
    # optional: run training directly (not used in SLURM pipeline)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=config.MODEL_NAME)
    args = parser.parse_args()
    model_name = args.model_name

    model_cfg = config.MODEL_CONFIGS[model_name]

    net = Autoencoder(
        latent_dim=model_cfg["latent_dim"],
        base_channels=model_cfg["base_channels"]
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    
    try:
        train_loader, val_loader = get_dataloaders(config.DATA_ROOT, config.VALIDATION_SIZE, config.BATCH_SIZE)
        history = train_model(
            net,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            config.NUM_EPOCHS,
            model_name=model_name,
            log_dir=config.LOGS_DIR
        )
    except FileNotFoundError as e:
        print(e)
