import os
import csv

import config
from evaluate import evaluate_model


def main():
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    summary_path = os.path.join(config.EVAL_DIR, "evaluation_summary.csv")

    fieldnames = ["model_name", "checkpoint_path", "mean_mse", "psnr"]
    rows = []

    # loop over all defined models
    for model_name in config.MODEL_CONFIGS.keys():
        print(f"=== Evaluating {model_name} ===")
        try:
            result = evaluate_model(model_name=model_name, checkpoint_path=None, num_images=16)
            rows.append(result)
        except FileNotFoundError as e:
            print(f"  Skipping {model_name}: {e}")

    if not rows:
        print("No models evaluated (no checkpoints found). Nothing to write.")
        return

    # write CSV summary
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
