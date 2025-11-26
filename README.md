🧠 Autoencoder for Face Reconstruction (FFHQ Dataset)

This project implements and compares 10 convolutional autoencoder architectures trained on a 256×256 subset of the FFHQ face dataset.
The goal is to understand how changes in latent dimensionality and channel width affect reconstruction quality.

All models were trained on a CUDA GPU using PyTorch, and evaluated using both quantitative metrics and visual reconstructions.

🚀 Features

10 Autoencoder variants (different latent sizes + channel widths)

GPU-accelerated training (CUDA)

Training/validation split: first 1,000 images = validation

Loss metric: MSE

Evaluation metrics:

Mean MSE

PSNR

Visual comparisons: “Before vs After”

Loss curve plots

Training history saved for each model

Clean, well-structured codebase

🏗 Project Structure
autoencoder_project/
│
├── config.py              # all hyperparameters + model variants
├── model.py               # convolutional autoencoder class
├── train.py               # training loop + dataloaders + history logging
├── main.py                # trains a single model end-to-end
├── evaluate.py            # evaluate a model + compute PSNR + save image grids
├── evaluate_all.py        # evaluate all models and export CSV summary
├── plot_losses.py         # generate loss curves from logs
│
├── models/                # saved checkpoints (.pt)
├── logs/                  # training histories
├── eval_outputs/          # before/after images + evaluation_summary.csv
├── loss_plots/            # loss curves for each model
│
└── README.md

🔧 Installation & Setup
git clone https://github.com/YOUR_USERNAME/autoencoder-faces.git
cd autoencoder-faces
pip install -r requirements.txt   # (optional, if you create one)


This project is written for Python 3.8+ and PyTorch with CUDA.

🏋️ Training a Model

Train any model by name:

python main.py --model-name ae_latent_64


All checkpoints are saved automatically into:

models/
logs/

📊 Evaluating a Model

Run evaluation + visual outputs:

python evaluate.py --model-name ae_latent_64


This will produce:

a grid image of 16 original/reconstructed pairs

printed metrics (MSE + PSNR)

Visuals are stored in:

eval_outputs/

📈 Evaluate All Models

To evaluate all 10 models at once:

python evaluate_all.py


This generates:

eval_outputs/evaluation_summary.csv

📉 Plot Loss Curves
python plot_losses.py


Outputs go to:

loss_plots/

📝 Results (Summary)

Best performing model:

Model name	MSE	PSNR (dB)
ae_wide_64	0.0284	15.47

Worst performing:

Model name	MSE	PSNR (dB)
ae_latent_4	0.1492	8.26

Reconstruction quality strongly correlates with:

wider networks → better detail

larger latent dimensions → fewer artifacts

tiny latent spaces → blurry, low-detail reconstructions

📄 Project Report

A full academic-style PDF explaining the architecture, experiments, and results is included.

👥 Authors

Yonatan Rosh
Uri Ben Dor
Reshit Carmel
