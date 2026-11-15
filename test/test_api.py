"""
Тесты API Doc Assistant через requests.
Перед запуском поднимите сервис:
    uvicorn src.app:app            (локально)
    docker compose up -d           (в контейнере)
Запуск:
    python test/test_api.py
"""
import os
import sys
import requests

BASE_URL = (
    sys.argv[1]
    if len(sys.argv) > 1 and sys.argv[1].startswith("http")
    else os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
)


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("GET /health:", r.status_code, r.json())
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask():
    r = requests.post(f"{BASE_URL}/ask",
                      json={"question": "What happened with the interest rates decided by the central bank?", "k": 3})
    data = r.json()
    print("POST /ask:", r.status_code)
    print("  answer:", data.get("answer"))
    print("  sources:", len(data.get("sources", [])))
    assert r.status_code == 200
    assert "answer" in data and "sources" in data
    assert len(data["sources"]) <= 3


def test_control_question():
    # Контрольный вопрос не по теме базы — источников быть не должно
    r = requests.post(f"{BASE_URL}/ask",
                      json={"question": "Who won the UEFA Champions League football tournament in 2024?", "k": 3})
    print("POST /ask (контрольный):", r.status_code, r.json().get("answer"))
    assert r.status_code == 200


def test_empty_question():
    r = requests.post(f"{BASE_URL}/ask", json={"question": "   ", "k": 3})
    print("POST /ask (пустой):", r.status_code, r.json())
    assert r.status_code == 400
    assert r.json()["detail"] == "Пустой вопрос"


def test_invalid_body():
    # Валидация Pydantic: k должно быть числом
    r = requests.post(f"{BASE_URL}/ask", json={"question": "test", "k": "abc"})
    print("POST /ask (k='abc'):", r.status_code)
    assert r.status_code == 422


if __name__ == "__main__":
    tests = [test_health, test_ask, test_control_question, test_empty_question, test_invalid_body]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}\n")
        except AssertionError as e:
            failed += 1
            print(f"❌ {t.__name__} {e}\n")
    print(f"Итого: {len(tests) - failed}/{len(tests)} тестов пройдено")
    sys.exit(1 if failed else 0)
