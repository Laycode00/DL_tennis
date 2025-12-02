# src/train.py

import argparse
import numpy as np
import torch
import yaml

from .datasets import load_and_prepare_data, get_dataloaders
from .models import LSTMModel, GRUModel
from .utils import train_and_evaluate, report_and_visualize, device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tennis impact detection experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp_01.yaml",
        help="Path to the experiment config file (.yaml)",
    )
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # ----------------------------
    # 1. Argument & Config
    # ----------------------------
    args = parse_args()
    config = load_config(args.config)

    print(f"Loaded config from: {args.config}")
    print(f"Experiment name: {config.get('experiment_name', 'N/A')}")
    print(f"Using device: {device}")

    # ----------------------------
    # 2. Seed
    # ----------------------------
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----------------------------
    # 3. Data & Training Settings
    # ----------------------------
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    file_path = data_cfg["file_path"]
    time_steps = data_cfg["time_steps"]
    features = data_cfg["features"]

    batch_size = train_cfg["batch_size"]
    num_epochs = train_cfg["num_epochs"]
    patience = train_cfg["patience"]
    learning_rate = train_cfg["learning_rate"]

    hidden_size = model_cfg["hidden_size"]
    num_layers = model_cfg["num_layers"]
    dropout = model_cfg["dropout"]
    models_to_run = model_cfg.get("models_to_run", ["LSTM", "GRU"])

    # ----------------------------
    # 4. Data Loading
    # ----------------------------
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        file_path=file_path,
        time_steps=time_steps,
        seed=seed,
    )
    train_loader, test_loader = get_dataloaders(
        X_train, X_test, y_train, y_test,
        batch_size=batch_size,
    )

    # ----------------------------
    # 5. Model Definitions
    # ----------------------------
    models = {}

    if "LSTM" in models_to_run:
        models["LSTM"] = LSTMModel(
            input_size=features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    if "GRU" in models_to_run:
        models["GRU"] = GRUModel(
            input_size=features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    if not models:
        raise ValueError("No valid models_to_run specified in config.")

    # ----------------------------
    # 6. Train & Evaluate
    # ----------------------------
    results = {}
    for model_name, model in models.items():
        print(f"\n=== Training and evaluating {model_name} model ===")
        acc, y_true, y_pred, y_scores = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            num_epochs=num_epochs,
            patience=patience,
            learning_rate=learning_rate,  # ← utils 쪽에서 받아서 사용
        )
        results[model_name] = {
            "accuracy": acc,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_scores": y_scores,
        }
        print(f"{model_name} Test Accuracy: {acc:.4f}\n")

        report_and_visualize(
            model_name,
            y_true,
            y_pred,
            y_scores,
        )


if __name__ == "__main__":
    main()



# run # python -m src.train --config configs/exp_01.yaml
