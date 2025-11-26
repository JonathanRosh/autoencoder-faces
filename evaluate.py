import os
import math
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

import config
from model import Autoencoder
from train import get_dataloaders, DEVICE


def denormalize(t):
    """Map [-1, 1] -> [0, 1] for saving images."""
    return t * 0.5 + 0.5


def find_latest_checkpoint(model_name):
    """Return latest checkpoint path for the given model_name, or None."""
    prefix = model_name + "_"
    candidates = []

    for f in os.listdir(config.MODELS_DIR):
        if not (f.endswith(".pt") and f.startswith(prefix)):
            continue
        if len(f) > len(prefix) and f[len(prefix)].isdigit():
            candidates.append(os.path.join(config.MODELS_DIR, f))

    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def evaluate_model(model_name, checkpoint_path=None, num_images=16):
    """
    Load a trained model, compute mean MSE + PSNR on validation set,
    and save a before/after image grid.
    """
    # checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(model_name)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found for model {model_name} in {config.MODELS_DIR}")
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        print(f"Using checkpoint: {checkpoint_path}")

    # model
    model_cfg = config.MODEL_CONFIGS[model_name]
    net = Autoencoder(
        latent_dim=model_cfg["latent_dim"],
        base_channels=model_cfg["base_channels"]
    ).to(DEVICE)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    net.load_state_dict(state_dict)
    net.eval()

    # validation loader
    _, val_loader = get_dataloaders(config.DATA_ROOT, config.VALIDATION_SIZE, config.BATCH_SIZE)

    mse_loss = nn.MSELoss(reduction="none")
    collected_orig, collected_recon = [], []
    all_mse = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            recon = net(batch)

            # per-image MSE
            sq_err = mse_loss(recon, batch)          # B x C x H x W
            per_img_mse = sq_err.view(sq_err.size(0), -1).mean(dim=1)
            all_mse.append(per_img_mse.cpu())

            # collect samples for visual comparison
            for orig_img, recon_img in zip(batch, recon):
                if len(collected_orig) >= num_images:
                    break
                collected_orig.append(orig_img.cpu())
                collected_recon.append(recon_img.cpu())
            if len(collected_orig) >= num_images:
                break

    if not all_mse:
        print("No images found in validation loader.")
        return

    all_mse = torch.cat(all_mse)
    mean_mse = all_mse.mean().item()
    psnr = 10.0 * math.log10(1.0 / mean_mse) if mean_mse > 0 else float("inf")

    print(f"Evaluation results for {model_name}:")
    print(f"  Mean MSE:  {mean_mse:.6f}")
    print(f"  PSNR:      {psnr:.2f} dB")

    # save before/after grid
    os.makedirs(config.EVAL_DIR, exist_ok=True)

    images_for_grid = []
    for o, r in zip(collected_orig, collected_recon):
        images_for_grid.append(denormalize(o))
        images_for_grid.append(denormalize(r))

    grid = make_grid(images_for_grid, nrow=2)
    out_name = f"{model_name}_before_after.png"
    out_path = os.path.join(config.EVAL_DIR, out_name)
    save_image(grid, out_path)
    print(f"Saved before/after image grid to {out_path}")

    return {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "mean_mse": mean_mse,
        "psnr": psnr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=config.MODEL_NAME)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-images", type=int, default=16)
    args = parser.parse_args()

    evaluate_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        num_images=args.num_images
    )
