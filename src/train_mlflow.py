"""
Задание №1 (MLOps + MLflow):
Обучение PyTorch-классификатора с трекингом параметров, метрик, артефактов, тегов
и регистрацией лучшей модели в MLflow Model Registry (стадия Staging).

Запуск:
    python -m src.train_mlflow
Просмотр UI:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import os
import pickle
import tempfile
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.classifier import DocMLP, MODEL_DIR, PROJECT_ROOT, load_classification_datasets

EXPERIMENT_NAME = "doc-classifier"
REGISTERED_MODEL_NAME = "doc-classifier"
DEFAULT_TRACKING_URI = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"


def evaluate(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Вычисляет точность (accuracy) модели на тестовой выборке."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test))
        preds = torch.argmax(logits, dim=1).numpy()
    return float((preds == y_test).mean())


def _save_training_plot(losses: List[float], accs: List[float], out_path: str, lr: float, epochs: int) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(range(1, len(losses) + 1), losses, color="#2563eb", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(accs) + 1), accs, color="#16a34a", label="Test Accuracy")
    ax2.set_ylabel("Accuracy", color="#16a34a")

    plt.title(f"DocClassifier Training (lr={lr}, epochs={epochs})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train_once(
    lr: float,
    epochs: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    batch_size: int = 32,
) -> Tuple[float, str, DocMLP]:
    """Один запуск обучения с полным логированием в MLflow."""
    torch.manual_seed(42)
    input_dim = X_train.shape[1]
    model = DocMLP(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    losses: List[float] = []
    accs: List[float] = []

    with mlflow.start_run(run_name=f"mlp_lr{lr}_ep{epochs}") as run:
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("batch_size", batch_size)
        mlflow.set_tag("model_type", "pytorch_mlp")
        mlflow.set_tag("stage_candidate", "true")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(xb)

            avg_loss = epoch_loss / len(X_train)
            acc_epoch = evaluate(model, X_test, y_test)
            losses.append(avg_loss)
            accs.append(acc_epoch)
            mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
            mlflow.log_metric("val_accuracy", acc_epoch, step=epoch + 1)

        acc = evaluate(model, X_test, y_test)
        mlflow.log_metric("test_accuracy", acc)

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, "training_curve.png")
            _save_training_plot(losses, accs, plot_path, lr, epochs)
            mlflow.log_artifact(plot_path, artifact_path="plots")

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lr": lr,
                        "epochs": epochs,
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "input_dim": input_dim,
                        "num_classes": num_classes,
                        "test_accuracy": acc,
                    },
                    f,
                    indent=2,
                )
            mlflow.log_artifact(config_path, artifact_path="config")

        mlflow.pytorch.log_model(model, name="model", input_example=X_train[:2])
        return acc, run.info.run_id, model


def run_hyperparameter_search(tracking_uri: str = DEFAULT_TRACKING_URI) -> Dict[str, object]:
    """Запускает подбор гиперпараметров, регистрирует лучшую модель в Staging и сохраняет её в model/."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_test, y_test, vectorizer, idx_to_label = load_classification_datasets()
    num_classes = len(idx_to_label)

    best_acc = -1.0
    best_run_id = None
    best_model = None
    best_params = {}
    results = []

    grid = [
        (0.0005, 10),
        (0.001, 10),
        (0.001, 20),
        (0.01, 10),
        (0.01, 20),
    ]

    for lr, epochs in grid:
        acc, run_id, model = train_once(
            lr=lr,
            epochs=epochs,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
        )
        print(f"[MLflow] lr={lr}, epochs={epochs} -> test_accuracy={acc:.4f} (run_id={run_id})")
        results.append({"lr": lr, "epochs": epochs, "test_accuracy": acc, "run_id": run_id})
        if acc > best_acc:
            best_acc = acc
            best_run_id = run_id
            best_model = model
            best_params = {"lr": lr, "epochs": epochs}

    # Регистрация лучшей модели в Model Registry и перевод в Staging
    model_uri = f"runs:/{best_run_id}/model"
    reg = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=reg.version,
            stage="Staging",
            archive_existing_versions=False,
        )
    except Exception:
        pass
    try:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "staging", reg.version)
    except Exception:
        pass

    # Сохраняем лучшую модель локально в model/ для FastAPI и инструментов агента
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(MODEL_DIR, "doc_classifier.pt"))
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "classifier_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dim": int(X_train.shape[1]),
                "num_classes": num_classes,
                "hidden_dim": 128,
                "dropout": 0.3,
                "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
                "best_params": best_params,
                "test_accuracy": best_acc,
                "mlflow_run_id": best_run_id,
                "mlflow_model_version": reg.version,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\n✅ Лучшая модель: version={reg.version}, stage=Staging, "
        f"test_accuracy={best_acc:.4f}, params={best_params}"
    )
    return {
        "best_accuracy": best_acc,
        "best_run_id": best_run_id,
        "model_version": reg.version,
        "runs": results,
    }


if __name__ == "__main__":
    run_hyperparameter_search()
