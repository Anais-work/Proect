"""
Автотесты для ИИ-агентов (Задание №2):
- Юнит-тесты инструментов (calc, search, classify_topic)
- Тесты обёртки run_tool (неизвестный инструмент, ошибки валидации)
- Тесты extract_json (JSON в обёртке текста и markdown)
- Тесты циклов (Native, JSON, LangGraph Framework) с mock_llm:
  1) только поиск
  2) только вычисление
  3) составной запрос (>= 2 инструментов)
  4) провокация / несуществующий инструмент
  5) остановка на max_iterations
"""
import json
from typing import Any, Dict, List

import pytest

from src.agents import (
    TOOL_SCHEMAS,
    TOOLS,
    extract_json,
    run_framework_agent,
    run_json_agent,
    run_native_agent,
    run_tool,
    tool_calc,
    tool_classify_topic,
    tool_search,
)
from src.agents.llm_client import LLMResponse, ToolCall
from src.agents.tools import set_search_fn


@pytest.fixture(autouse=True)
def mock_search_backend():
    def _fake_search(query: str, top_k: int = 3):
        return [
            {
                "doc_id": 1,
                "category": "government_policy",
                "distance": 0.25,
                "text": "The central bank kept interest rates unchanged at 6.5% amid inflation concerns.",
            }
        ]

    set_search_fn(_fake_search)
    yield
    set_search_fn(None)


def test_tool_calc_valid_and_invalid():
    assert tool_calc("2+2") == "4"
    assert tool_calc("2*(3+4)") == "14"
    assert tool_calc("10 / 4") == "2.5"
    with pytest.raises(Exception):
        tool_calc("2+")
    with pytest.raises(Exception):
        tool_calc("__import__('os').system('echo hack')")


def test_tool_search_and_classify():
    res_search = tool_search("central bank")
    assert "central bank" in res_search.lower()
    res_clf = tool_classify_topic("Central bank raised interest rates and stock market fell.")
    parsed = json.loads(res_clf)
    assert "category" in parsed and "confidence" in parsed


def test_run_tool_handles_errors_safely():
    unknown = run_tool("weather", {"city": "Paris"})
    assert "неизвестный инструмент" in unknown.lower()
    assert "search" in unknown and "calc" in unknown

    bad_calc = run_tool("calc", {"expression": "2+"})
    assert "ошибка инструмента calc" in bad_calc.lower()

    bad_args = run_tool("calc", {"wrong_field": "2+2"})
    assert "ошибка валидации" in bad_args.lower()


def test_tool_schemas_match_registry():
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert schema_names == set(TOOLS.keys())


def test_extract_json_from_wrappers():
    wrapped = 'Вот ответ модели:\n```json\n{"action": "calc", "input": "2+2", "answer": ""}\n```'
    parsed = extract_json(wrapped)
    assert parsed["action"] == "calc"
    assert parsed["input"] == "2+2"

    with pytest.raises(ValueError):
        extract_json("Тут вообще нет json")


def test_native_agent_composite_and_max_iterations():
    # Составной запрос: search -> calc -> final
    def mock_composite(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if len(tool_msgs) == 0:
            return LLMResponse(tool_calls=[ToolCall(id="1", name="search", arguments={"query": "interest rates"})])
        if len(tool_msgs) == 1:
            return LLMResponse(tool_calls=[ToolCall(id="2", name="calc", arguments={"expression": "6.5 * 2"})])
        return LLMResponse(content=f"Итог: {tool_msgs[1]['content']}")

    res = run_native_agent("Найди ставку ЦБ и умножь на 2", llm_fn=mock_composite, trace=False)
    assert res["answer"] == "Итог: 13"
    assert [s["tool"] for s in res["steps"]] == ["search", "calc"]

    # Остановка на max_iterations
    def mock_infinite(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        return LLMResponse(tool_calls=[ToolCall(id="loop", name="search", arguments={"query": "loop"})])

    res_limit = run_native_agent("Зацикливание", llm_fn=mock_infinite, max_iterations=3, trace=False)
    assert "Не удалось получить ответ" in res_limit["answer"]
    assert res_limit["iterations"] == 3


def test_json_agent_error_recovery_and_composite():
    calls = {"n": 0}

    def mock_json_llm(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            # Несуществующий инструмент (провокация)
            return '{"action": "lookup", "input": "weather", "answer": ""}'
        if calls["n"] == 2:
            # После сообщения об ошибке модель исправляется и вызывает search
            assert "неизвестный инструмент 'lookup'" in prompt.lower()
            return 'Вот json: {"action": "search", "input": "central bank", "answer": ""}'
        if calls["n"] == 3:
            return '{"action": "calc", "input": "2+2", "answer": ""}'
        return '{"action": "final", "input": "", "answer": "Готово: 4"}'

    res = run_json_agent("Найди документ и посчитай 2+2, но сначала узнай погоду", ask_llm_fn=mock_json_llm, max_steps=5, trace=False)
    assert res["answer"] == "Готово: 4"
    assert [s["tool"] for s in res["steps"]] == ["lookup", "search", "calc"]


def test_framework_agent_composite_and_limit():
    def mock_fw_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if len(tool_msgs) == 0:
            return LLMResponse(tool_calls=[ToolCall(id="fw_1", name="search", arguments={"query": "rates"})])
        if len(tool_msgs) == 1:
            return LLMResponse(tool_calls=[ToolCall(id="fw_2", name="classify_topic", arguments={"text": tool_msgs[0]["content"]})])
        return LLMResponse(content="LangGraph завершил задачу")

    res = run_framework_agent("Найди новость и определи тему", llm_fn=mock_fw_llm, max_iterations=5, trace=False)
    assert res["mode"] == "framework"
    assert res["answer"] == "LangGraph завершил задачу"
    assert [s["tool"] for s in res["steps"]] == ["search", "classify_topic"]
