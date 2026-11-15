"""
Пакет ИИ-агентов (Задание №2):
- tools: 3 инструмента (search, calc, classify_topic), схемы и безопасная обёртка run_tool
- native_agent: ручной цикл с tool_calls / JSON Schema
- json_agent: ручной цикл с SYSTEM-промптом и extract_json
- framework_agent: граф-агент на базе LangGraph
"""
from src.agents.tools import TOOLS, TOOL_SCHEMAS, run_tool, tool_calc, tool_classify_topic, tool_search
from src.agents.native_agent import run_native_agent
from src.agents.json_agent import extract_json, run_json_agent
from src.agents.framework_agent import run_framework_agent

__all__ = [
    "TOOLS",
    "TOOL_SCHEMAS",
    "run_tool",
    "tool_search",
    "tool_calc",
    "tool_classify_topic",
    "run_native_agent",
    "extract_json",
    "run_json_agent",
    "run_framework_agent",
]
