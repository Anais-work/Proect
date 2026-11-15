"""
Автотесты классификатора, MLflow-артефактов и FastAPI-эндпоинтов (/health, /classify, /agent).
"""
import os

from fastapi.testclient import TestClient

from src.app import app
from src.classifier import MODEL_DIR, get_classifier

client = TestClient(app)


def test_classifier_artifacts_exist_and_predict():
    assert os.path.exists(os.path.join(MODEL_DIR, "doc_classifier.pt"))
    assert os.path.exists(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    assert os.path.exists(os.path.join(MODEL_DIR, "classifier_meta.json"))

    clf = get_classifier()
    res = clf.predict("Sensex and Nifty surged as banking stocks rallied on strong quarterly earnings.")
    assert "category" in res
    assert 0.0 <= res["confidence"] <= 1.0


def test_fastapi_health_and_classify():
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}

    r_cls = client.post("/classify", json={"text": "Reserve Bank kept repo rate unchanged."})
    assert r_cls.status_code == 200
    data = r_cls.json()
    assert "category" in data and "probabilities" in data

    r_empty = client.post("/classify", json={"text": "   "})
    assert r_empty.status_code == 400


def test_fastapi_agent_validation():
    r_empty = client.post("/agent", json={"question": " ", "mode": "native"})
    assert r_empty.status_code == 400

    r_bad_mode = client.post("/agent", json={"question": "test", "mode": "unknown"})
    assert r_bad_mode.status_code == 422


def test_fastapi_calc():
    r_ok = client.post("/calc", json={"expression": "125 * 4 + 50"})
    assert r_ok.status_code == 200
    assert r_ok.json() == {"expression": "125 * 4 + 50", "result": "550"}

    r_bad = client.post("/calc", json={"expression": "2+"})
    assert r_bad.status_code == 400

    r_empty = client.post("/calc", json={"expression": "   "})
    assert r_empty.status_code == 400
