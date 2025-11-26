import os
import torch
import matplotlib.pyplot as plt

LOGS_DIR = "./logs"
OUT_DIR = "./loss_plots"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_history(name, path):
    """Plot train/val loss vs epoch for a single history file."""
    hist = torch.load(path, map_location="cpu")
    train = hist["train_loss"]
    val = hist["val_loss"]

    plt.figure(figsize=(8, 4))
    plt.plot(train, label="Train Loss")
    plt.plot(val, label="Val Loss")
    plt.title(f"Loss Curves – {name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(OUT_DIR, f"{name}_loss.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    """Loop over all history files and plot their curves."""
    for filename in os.listdir(LOGS_DIR):
        if filename.endswith(".pt"):
            path = os.path.join(LOGS_DIR, filename)
            name = filename.split("_history")[0]
            plot_history(name, path)


if __name__ == "__main__":
    main()
