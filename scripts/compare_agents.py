"""
Сравнение 3 подходов ИИ-агентов (Native, JSON, LangGraph Framework) на составном запросе,
где агент самостоятельно вызывает >= 2 инструментов.

Запуск с mock LLM (детерминированно, без внешнего API):
    python -m scripts.compare_agents --mock

Запуск с реальным GigaChat API (требует .env):
    python -m scripts.compare_agents
"""
import argparse
import json
import time
from typing import Any, Dict, List

from src.agents import run_framework_agent, run_json_agent, run_native_agent
from src.agents.llm_client import LLMResponse, ToolCall
from src.classifier import get_classifier
from src.rag import get_rag

COMPOSITE_QUERY = (
    "Найди в базе новостей информацию о ставке центрального банка, "
    "определи тему найденного текста и посчитай 125 * 4 + 50."
)


def _make_mock_native_llm():
    def mock_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if len(tool_msgs) == 0:
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="call_1", name="search", arguments={"query": "central bank interest rates"})
                ]
            )
        if len(tool_msgs) == 1:
            snippet = tool_msgs[0]["content"][:200]
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="call_2", name="classify_topic", arguments={"text": snippet}),
                    ToolCall(id="call_3", name="calc", arguments={"expression": "125 * 4 + 50"}),
                ]
            )
        return LLMResponse(
            content=(
                f"Найдена новость (результат search), определена тема ({tool_msgs[1]['content']}), "
                f"вычислено выражение 125 * 4 + 50 = {tool_msgs[2]['content']}."
            )
        )

    return mock_llm


def _make_mock_json_llm():
    def mock_ask(prompt: str) -> str:
        if "Действие: search" not in prompt:
            return json.dumps({"action": "search", "input": "central bank interest rates", "answer": ""})
        if "Действие: classify_topic" not in prompt:
            return json.dumps({"action": "classify_topic", "input": "Central bank decided on interest rates", "answer": ""})
        if "Действие: calc" not in prompt:
            return json.dumps({"action": "calc", "input": "125 * 4 + 50", "answer": ""})
        return json.dumps(
            {
                "action": "final",
                "input": "",
                "answer": "Поиск по ЦБ выполнен, тема определена, 125 * 4 + 50 = 550.",
            },
            ensure_ascii=False,
        )

    return mock_ask


def compare(use_mock: bool = False, query: str = COMPOSITE_QUERY) -> List[Dict[str, Any]]:
    # Предварительно прогреваем BERT/FAISS и классификатор, чтобы время холодной загрузки
    # не искажало сравнение скорости трёх архитектур агентов.
    get_rag()
    get_classifier()

    print(f"Запрос: {query}\n" + "=" * 75)
    summary = []

    runners = [
        ("native", lambda: run_native_agent(query, llm_fn=_make_mock_native_llm() if use_mock else None)),
        ("json", lambda: run_json_agent(query, ask_llm_fn=_make_mock_json_llm() if use_mock else None)),
        ("framework", lambda: run_framework_agent(query, llm_fn=_make_mock_native_llm() if use_mock else None)),
    ]

    for name, fn in runners:
        print(f"\n--- Подход: {name.upper()} ---")
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        tools_used = [s.get("tool") for s in res.get("steps", []) if s.get("tool")]
        print(f"Ответ: {res['answer']}")
        print(f"Итераций: {res['iterations']}, инструментов вызвано: {len(tools_used)} ({tools_used}), время: {dt:.3f}s")
        summary.append(
            {
                "mode": name,
                "iterations": res["iterations"],
                "tools_called": tools_used,
                "elapsed_sec": round(dt, 4),
                "answer": res["answer"],
            }
        )

    print("\n" + "=" * 75)
    print("СВОДКА СРАВНЕНИЯ:")
    for row in summary:
        print(
            f"  {row['mode']:10s} | iterations={row['iterations']} | "
            f"tools={row['tools_called']} | time={row['elapsed_sec']}s"
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение 3 подходов ИИ-агентов")
    parser.add_argument("--mock", action="store_true", help="Использовать mock LLM без вызова GigaChat API")
    parser.add_argument("-q", "--query", type=str, default=COMPOSITE_QUERY, help="Составной запрос")
    args = parser.parse_args()
    compare(use_mock=args.mock, query=args.query)
