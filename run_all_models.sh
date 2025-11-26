#!/bin/bash

MODELS=(
  "ae_baseline"
  "ae_latent_64"
  "ae_latent_32"
  "ae_latent_16"
  "ae_latent_8"
  "ae_wide_32"
  "ae_wide_64"
  "ae_wide_32_latent_64"
  "ae_wide_64_latent_16"
  "ae_latent_4"
)

for NAME in "${MODELS[@]}"; do
  echo "Submitting job for $NAME"
  sbatch main_slurm.sh "$NAME"
done
