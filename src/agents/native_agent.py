"""
Вариант 1 (подход Native):
Ручной цикл агента с использованием структурированных описаний инструментов (TOOL_SCHEMAS),
истории диалога `messages`, `tool_call_id` и лимита `max_iterations`.
"""
from typing import Any, Callable, Dict, List, Optional

from src.agents.llm_client import LLMResponse, get_gigachat_client
from src.agents.tools import TOOL_SCHEMAS, run_tool

SYSTEM_PROMPT = (
    "Ты бизнес-ассистент с доступом к инструментам: search, calc, classify_topic. "
    "Вызывай инструменты только когда они действительно нужны. "
    "Когда соберёшь достаточно данных, дай чёткий финальный ответ без вызова инструментов."
)


def _default_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
    return get_gigachat_client().chat(messages=messages, tools=tools)


def run_native_agent(
    user_query: str,
    llm_fn: Optional[Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], LLMResponse]] = None,
    max_iterations: int = 5,
    trace: bool = True,
) -> Dict[str, Any]:
    """Запускает Native-агента и возвращает ответ + лог шагов."""
    llm = llm_fn or _default_llm
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    steps: List[Dict[str, Any]] = []

    for step in range(max_iterations):
        response = llm(messages, TOOL_SCHEMAS)

        if not response.tool_calls:
            return {
                "mode": "native",
                "answer": response.content,
                "steps": steps,
                "iterations": step + 1,
            }

        # Сохраняем запрос ассистента на вызов инструментов в историю
        messages.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in response.tool_calls
                ],
            }
        )

        for call in response.tool_calls:
            result = run_tool(call.name, call.arguments)
            steps.append(
                {
                    "step": step,
                    "tool": call.name,
                    "args": call.arguments,
                    "result": result,
                }
            )
            if trace:
                print(f"[native step {step}] tool={call.name} args={call.arguments} -> {result[:120]}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result,
                }
            )

    return {
        "mode": "native",
        "answer": "Не удалось получить ответ за отведенное число шагов.",
        "steps": steps,
        "iterations": max_iterations,
    }
