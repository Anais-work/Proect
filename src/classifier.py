"""
PyTorch MLP-классификатор бизнес-новостей по категориям.
"""
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


class DocMLP(nn.Module):
    """Многослойный перцептрон для классификации TF-IDF векторов документов."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_classification_datasets(
    data_dir: str = DATA_DIR, max_features: int = 5000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer, Dict[int, str]]:
    """Загружает train/test CSV, строит TF-IDF признаки и кодирует метки классов."""
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv")).dropna(subset=["text", "category"])
    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv")).dropna(subset=["text", "category"])

    classes = sorted(train_df["category"].astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train = vectorizer.fit_transform(train_df["text"].astype(str)).toarray().astype(np.float32)
    X_test = vectorizer.transform(test_df["text"].astype(str)).toarray().astype(np.float32)

    y_train = train_df["category"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64)
    y_test = test_df["category"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64)

    return X_train, y_train, X_test, y_test, vectorizer, idx_to_label


class TopicClassifier:
    """Обёртка для инференса сохранённого PyTorch-классификатора."""

    def __init__(self, model: DocMLP, vectorizer: TfidfVectorizer, idx_to_label: Dict[int, str]):
        self.model = model
        self.model.eval()
        self.vectorizer = vectorizer
        self.idx_to_label = idx_to_label

    @classmethod
    def load(cls, model_dir: str = MODEL_DIR) -> "TopicClassifier":
        weights_path = os.path.join(model_dir, "doc_classifier.pt")
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        meta_path = os.path.join(model_dir, "classifier_meta.json")

        if not (os.path.exists(weights_path) and os.path.exists(vec_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"Модель классификатора не найдена в {model_dir}. "
                "Запустите обучение: python -m src.train_mlflow"
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)

        idx_to_label = {int(k): v for k, v in meta["idx_to_label"].items()}
        model = DocMLP(
            input_dim=meta["input_dim"],
            num_classes=meta["num_classes"],
            hidden_dim=meta["hidden_dim"],
            dropout=meta.get("dropout", 0.3),
        )
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return cls(model=model, vectorizer=vectorizer, idx_to_label=idx_to_label)

    def predict(self, text: str) -> Dict[str, object]:
        """Возвращает предсказанную категорию и уверенность модели."""
        vec = self.vectorizer.transform([text]).toarray().astype(np.float32)
        tensor = torch.from_numpy(vec)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            best_idx = int(torch.argmax(probs).item())
            confidence = float(probs[best_idx].item())

        return {
            "category": self.idx_to_label[best_idx],
            "confidence": round(confidence, 4),
            "probabilities": {
                self.idx_to_label[i]: round(float(probs[i].item()), 4)
                for i in range(len(self.idx_to_label))
            },
        }


_cached_classifier: Optional[TopicClassifier] = None


def get_classifier() -> TopicClassifier:
    """Ленивая загрузка и кэширование классификатора."""
    global _cached_classifier
    if _cached_classifier is None:
        _cached_classifier = TopicClassifier.load()
    return _cached_classifier
