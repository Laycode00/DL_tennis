# src/utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_fscore_support, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} "
                    f"out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0


def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    num_epochs: int = 50,
    patience: int = 3,
    learning_rate: float = 0.001,
):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= len(test_loader)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Val Loss: {val_loss:.4f}"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    model.load_state_dict(early_stopping.best_model_wts)

    # final evaluation
    model.eval()
    correct = 0
    total = 0
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()

            y_true.extend(y_batch.cpu().tolist())
            y_pred.extend(predicted.squeeze().cpu().tolist())
            y_scores.extend(outputs.squeeze().cpu().tolist())

            total += y_batch.size(0)
            correct += (predicted.squeeze() == y_batch).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy, y_true, y_pred, y_scores


def report_and_visualize(model_name, y_true, y_pred, y_scores):
    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=[0, 1]
    )
    disp.plot(cmap=plt.cm.Oranges)
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"{model_name} Performance:")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")

    # per-class precision/recall/F1
    precision_0 = (
        conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
        if (conf_matrix[0, 0] + conf_matrix[1, 0]) != 0 else 0
    )
    recall_0 = (
        conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        if (conf_matrix[0, 0] + conf_matrix[0, 1]) != 0 else 0
    )
    f1_0 = (
        2 * (precision_0 * recall_0) / (precision_0 + recall_0)
        if (precision_0 + recall_0) != 0 else 0
    )

    precision_1 = (
        conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
        if (conf_matrix[1, 1] + conf_matrix[0, 1]) != 0 else 0
    )
    recall_1 = (
        conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
        if (conf_matrix[1, 1] + conf_matrix[1, 0]) != 0 else 0
    )
    f1_1 = (
        2 * (precision_1 * recall_1) / (precision_1 + recall_1)
        if (precision_1 + recall_1) != 0 else 0
    )

    print(
        f"Class 0 - Precision: {precision_0:.4f}, "
        f"Recall: {recall_0:.4f}, F1: {f1_0:.4f}"
    )
    print(
        f"Class 1 - Precision: {precision_1:.4f}, "
        f"Recall: {recall_1:.4f}, F1: {f1_1:.4f}\n"
    )

    # ROC & AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # aggregated precision/recall/F1
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
    }

    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.title(f"{model_name} Precision, Recall, and F1-Score")
    plt.show()
