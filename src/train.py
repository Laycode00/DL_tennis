# src/train.py

import numpy as np
import torch

from .datasets import load_and_prepare_data, get_dataloaders
from .models import LSTMModel, GRUModel
from .utils import train_and_evaluate, report_and_visualize, device


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Using device: {device}")

    # config
    time_steps = 50
    features = 18
    batch_size = 256
    hidden_size = 256
    num_layers = 2
    dropout = 0.2
    num_epochs = 50
    patience = 3

    file_path = "data/sliced_data.csv"

    # data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        file_path=file_path,
        time_steps=time_steps,
        seed=seed,
    )
    train_loader, test_loader = get_dataloaders(
        X_train, X_test, y_train, y_test,
        batch_size=batch_size,
    )

    # model
    models = {
        "LSTM": LSTMModel(
            input_size=features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ),
        "GRU": GRUModel(
            input_size=features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ),
    }

    # train + test
    results = {}
    for model_name, model in models.items():
        print(f"\n=== Training and evaluating {model_name} model ===")
        acc, y_true, y_pred, y_scores = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            num_epochs=num_epochs,
            patience=patience,
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
