# Autoencoder for Face Reconstruction (FFHQ Dataset)

This project implements and compares 10 convolutional autoencoder architectures trained on a 256×256 subset of the FFHQ face dataset.  
The goal is to understand how changes in latent dimensionality and channel width affect reconstruction quality.

All models were trained on a CUDA GPU using PyTorch and evaluated using both quantitative metrics and visual reconstructions.

---

## 🚀 Features

- 10 Autoencoder variants (different latent sizes + channel widths)
- GPU-accelerated training (CUDA)
- Training/validation split: first **1,000 images = validation**
- Loss metric: **MSE**
- Evaluation metrics: **Mean MSE**, **PSNR**
- Visual comparisons: “Before vs After”
- Loss curve plots for each model
- Training history saved for every model
- Clean, modular project structure

---

## 📁 Project Structure

```
autoencoder_project/
│
├── config.py              # Hyperparameters + model configs
├── model.py               # Autoencoder architecture
├── train.py               # Training loop + dataloaders
├── main.py                # Train a single model end-to-end
├── evaluate.py            # Evaluate one model + save images
├── evaluate_all.py        # Evaluate all models + write summary CSV
├── plot_losses.py         # Generate loss curves from logs
│
├── models/                # Trained model checkpoints (.pt)
├── logs/                  # Training history files
├── eval_outputs/          # Before/after reconstructions + summary
├── loss_plots/            # Loss curve plots
│
└── README.md
```

---

## 🛠 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/autoencoder-faces.git
cd autoencoder-faces
```

(Optional) Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements:

- Python 3.8+
- PyTorch with CUDA enabled

---

## 🧠 Training a Model

Train a model variant:

```bash
python main.py --model-name ae_latent_64
```

Models are saved under:

```
models/
```

Training histories (for plotting loss curves) are saved under:

```
logs/
```

---

## 📊 Evaluating a Model

Evaluate one model:

```bash
python evaluate.py --model-name ae_latent_64
```

This produces:

- Mean MSE  
- PSNR  
- A before/after image grid  

Saved in:

```
eval_outputs/
```

---

## 📈 Evaluate All Models at Once

```bash
python evaluate_all.py
```

Creates:

```
eval_outputs/evaluation_summary.csv
```

---

## 📉 Plot Loss Curves

```bash
python plot_losses.py
```

Plots stored in:

```
loss_plots/
```

---

## ⭐ Results Summary

### Best Model (Highest PSNR)

| Model      | MSE     | PSNR (dB) |
|------------|---------|-----------|
| ae_wide_64 | 0.0284  | 15.47     |

### Worst Model (Tiny Latent)

| Model        | MSE     | PSNR (dB) |
|--------------|---------|-----------|
| ae_latent_4  | 0.1492  | 8.26      |

### Key Insights

- Larger latent spaces → significantly better reconstruction quality  
- Wider networks (more channels) → sharper and more detailed outputs  
- Ultra-compressed models → blurry faces + identity loss  
- Training loss > validation loss due to batch noise (BatchNorm)  

---

## 👤 Authors

- **Yonatan Rosh**
- **Uri Ben Dor**
- **Reshit Carmel**

---

## 📄 License

MIT License — feel free to use or modify.

