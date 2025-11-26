import os
from datetime import datetime
import argparse

import config
from train import get_dataloaders, train_model, DEVICE
from model import Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim


def load_model(input_path, model_name):
    """Create model with given config and load weights."""
    model_cfg = config.MODEL_CONFIGS[model_name]
    net = Autoencoder(
        latent_dim=model_cfg["latent_dim"],
        base_channels=model_cfg["base_channels"]
    ).to(DEVICE)
    net.load_state_dict(torch.load(input_path, map_location=DEVICE))
    return net.to(DEVICE)


def store_model(net, output_path):
    """Save model weights only."""
    torch.save(net.state_dict(), output_path)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=config.MODEL_NAME)
    args = parser.parse_args()
    model_name = args.model_name

    print(f"Using model config: {model_name}")
    model_cfg = config.MODEL_CONFIGS[model_name]

    # load or create model
    if config.input_model:
        print(f'Loading model {config.input_model}')
        net = load_model(os.path.join(config.MODELS_DIR, config.input_model), model_name)
    else:
        print('Creating a new model!')
        net = Autoencoder(
            latent_dim=model_cfg["latent_dim"],
            base_channels=model_cfg["base_channels"]
        ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    
    train_loader, val_loader = get_dataloaders(config.DATA_ROOT, config.VALIDATION_SIZE, config.BATCH_SIZE)

    # train and log history
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
    
    # save weights with model name + timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(config.MODELS_DIR, f'{model_name}_{timestamp}.pt')
    print(f'Storing model at {output_path}')
    store_model(net, output_path)


if __name__ == '__main__':
    main()
