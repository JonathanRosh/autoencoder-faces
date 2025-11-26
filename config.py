# Global paths and hyperparameters

MODELS_DIR = '/home/ML_courses/03683533_2025/yonatan_uri_reshit/autoencoder/models'
LOGS_DIR = '/home/ML_courses/03683533_2025/yonatan_uri_reshit/autoencoder/logs'
EVAL_DIR = '/home/ML_courses/03683533_2025/yonatan_uri_reshit/autoencoder/eval_outputs'

DATA_ROOT = "/home/ML_courses/03683533_2025/dataset"
# DATA_ROOT = "/home/ML_courses/03683533_2025/yonatan_uri_reshit/parsed_dataset"

BATCH_SIZE = 32
LEARNING_RATE = 0.0002
NUM_EPOCHS = 10
VALIDATION_SIZE = 1000
BETAS = (0.5, 0.999)

# default model (can be overridden by --model-name)
MODEL_NAME = "ae_latent_64"

# model_name -> (latent_dim, base_channels)
MODEL_CONFIGS = {
    "ae_baseline": {
        "latent_dim": 256,
        "base_channels": 16,
    },
    "ae_latent_64": {
        "latent_dim": 64,
        "base_channels": 16,
    },
    "ae_latent_32": {
        "latent_dim": 32,
        "base_channels": 16,
    },
    "ae_latent_16": {
        "latent_dim": 16,
        "base_channels": 16,
    },
    "ae_latent_8": {
        "latent_dim": 8,
        "base_channels": 16,
    },
    "ae_wide_32": {
        "latent_dim": 256,
        "base_channels": 32,
    },
    "ae_wide_64": {
        "latent_dim": 256,
        "base_channels": 64,
    },
    "ae_wide_32_latent_64": {
        "latent_dim": 64,
        "base_channels": 32,
    },
    "ae_wide_64_latent_16": {
        "latent_dim": 16,
        "base_channels": 64,
    },
    "ae_latent_4": {
        "latent_dim": 4,
        "base_channels": 16,
    },
}

# optional: load from an existing checkpoint
input_model = ''
output_model = 'auto_encoder1'
