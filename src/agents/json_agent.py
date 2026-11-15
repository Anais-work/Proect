"""
Вариант 2 (JSON-подход для любой LLM):
SYSTEM-промпт с контрактом JSON, парсер `extract_json`, валидация `action`,
обёртка `run_tool` и логирование сырого ответа `raw`.
"""
import json
import re
from typing import Any, Callable, Dict, List, Optional

from src.agents.llm_client import get_gigachat_client
from src.agents.tools import run_tool

SYSTEM_PROMPT = (
    "Ты агент с инструментами: search(query), calc(expression), classify_topic(text).\n"
    "База документов содержит англоязычные новости, поэтому для search формулируй query на английском "
    "(например, 'central bank interest rates').\n"
    "На каждом шаге верни ТОЛЬКО JSON без пояснений:\n"
    '{"action":"search|calc|classify_topic|final","input":"...","answer":"..."}\n'
    "Для action=final заполни answer, для остальных — input.\n"
    "Используй инструменты, пока не соберёшь ответ."
)


def extract_json(text: str) -> Dict[str, Any]:
    """Извлекает JSON-объект из ответа модели (включая markdown-обёртку)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("В ответе модели нет JSON")
    raw_json = match.group(0)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        parsed, _ = json.JSONDecoder().raw_decode(raw_json.lstrip())
    if not isinstance(parsed, dict):
        raise ValueError("Ожидался JSON-объект")
    return parsed


def _default_ask_llm(prompt: str) -> str:
    user_content = (
        prompt[len(SYSTEM_PROMPT):].lstrip("\n")
        if prompt.startswith(SYSTEM_PROMPT)
        else prompt
    )
    resp = get_gigachat_client().chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tools=None,
    )
    return resp.content


def run_json_agent(
    question: str,
    ask_llm_fn: Optional[Callable[[str], str]] = None,
    max_steps: int = 5,
    trace: bool = True,
) -> Dict[str, Any]:
    """Запускает JSON-агента и возвращает ответ + лог шагов."""
    ask_llm = ask_llm_fn or _default_ask_llm
    history = f"Вопрос: {question}\n"
    steps: List[Dict[str, Any]] = []

    for step_num in range(max_steps):
        raw = ask_llm(SYSTEM_PROMPT + "\n" + history)
        if trace:
            print(f"[json step {step_num}] raw={raw[:200]}")

        try:
            step = extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            err_msg = f"Ошибка: ответ не парсится как JSON ({e}). Верни чистый JSON."
            history += err_msg + "\n"
            steps.append({"step": step_num, "raw": raw, "error": err_msg})
            continue

        action = str(step.get("action", "")).strip().lower()
        if action == "final":
            raw_answer = step.get("answer")
            answer = (
                "Модель не дала ответ"
                if raw_answer is None or raw_answer == ""
                else str(raw_answer)
            )
            return {
                "mode": "json",
                "answer": answer,
                "steps": steps,
                "iterations": step_num + 1,
            }

        raw_input = step.get("input")
        input_value = "" if raw_input is None else str(raw_input)
        result = run_tool(action, input_value)
        steps.append(
            {
                "step": step_num,
                "raw": raw,
                "tool": action,
                "args": input_value,
                "result": result,
            }
        )
        if trace:
            print(f"[json step {step_num}] action={action} input={input_value} -> {result[:120]}")

        history += f"Действие: {action}({input_value}) -> {result}\n"

    return {
        "mode": "json",
        "answer": "Не удалось получить ответ за отведенное число шагов.",
        "steps": steps,
        "iterations": max_steps,
    }
