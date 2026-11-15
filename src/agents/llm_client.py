"""
Общий клиент GigaChat для агентов (поддержка обычного чата и function/tool calling).
"""
import base64
from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional
import uuid

import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


def _to_json_function_content(content: Any) -> str:
    """Гарантирует, что результат функции для GigaChat является JSON-объектом в виде строки."""
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    content_str = str(content)
    try:
        parsed = json.loads(content_str)
        if isinstance(parsed, dict):
            return content_str
    except (json.JSONDecodeError, TypeError):
        pass
    return json.dumps({"result": content_str}, ensure_ascii=False)


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _convert_messages_for_gigachat(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Конвертирует историю диалога из формата OpenAI (`tool_calls` + `role='tool'`)
    в формат GigaChat (`function_call` + `role='function'`), не изменяя исходный список.
    """
    converted: List[Dict[str, Any]] = []
    call_id_to_name: Dict[str, str] = {}
    i = 0

    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            for tc in tool_calls:
                tc_id = tc.get("id")
                tc_name = tc.get("function", {}).get("name", "")
                if tc_id and tc_name:
                    call_id_to_name[str(tc_id)] = tc_name

            if len(tool_calls) == 1:
                tc = tool_calls[0]
                fn = tc.get("function", {})
                giga_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "function_call": {
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", {}),
                    },
                }
                tc_id = tc.get("id")
                if tc_id and _is_valid_uuid(str(tc_id)):
                    giga_msg["functions_state_id"] = str(tc_id)
                converted.append(giga_msg)
                i += 1
                continue

            # Если в одном сообщении несколько tool_calls, разворачиваем их в пары
            # assistant(function_call) -> function, чтобы GigaChat принял всю цепочку.
            tool_results_by_id: Dict[str, Dict[str, Any]] = {}
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                t_msg = messages[j]
                t_id = str(t_msg.get("tool_call_id") or "")
                tool_results_by_id[t_id] = t_msg
                j += 1

            for idx, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                tc_id = str(tc.get("id") or "")
                giga_msg = {
                    "role": "assistant",
                    "content": (msg.get("content") or "") if idx == 0 else "",
                    "function_call": {
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", {}),
                    },
                }
                if tc_id and _is_valid_uuid(tc_id):
                    giga_msg["functions_state_id"] = tc_id
                converted.append(giga_msg)

                if tc_id in tool_results_by_id:
                    t_msg = tool_results_by_id[tc_id]
                    fn_name = t_msg.get("name") or fn.get("name", "")
                    converted.append(
                        {
                            "role": "function",
                            "name": fn_name,
                            "content": _to_json_function_content(t_msg.get("content", "")),
                        }
                    )
            i = j
            continue

        if role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            fn_name = msg.get("name") or call_id_to_name.get(call_id, "")
            converted.append(
                {
                    "role": "function",
                    "name": fn_name,
                    "content": _to_json_function_content(msg.get("content", "")),
                }
            )
            i += 1
            continue

        converted.append(dict(msg))
        i += 1

    return converted


class GigaChatClient:
    """Лёгкий клиент GigaChat API с кэшированием OAuth-токена."""

    def __init__(self) -> None:
        self._token: Optional[str] = None

    def get_token(self) -> Optional[str]:
        if self._token:
            return self._token
        client_id = os.getenv("GIGA_CLIENT_ID")
        client_secret = os.getenv("GIGA_CLIENT_SECRET")
        if not client_id or not client_secret:
            return None

        auth_b64 = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_b64}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = requests.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers=headers,
            data={"scope": os.getenv("GIGA_SCOPE", "GIGACHAT_API_PERS")},
            verify=False,
            timeout=15,
        )
        if resp.status_code == 200:
            self._token = resp.json().get("access_token")
            return self._token
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        token = self.get_token()
        if not token:
            raise RuntimeError("Не удалось получить токен GigaChat. Проверьте GIGA_CLIENT_ID / GIGA_CLIENT_SECRET в .env.")

        payload: Dict[str, Any] = {
            "model": os.getenv("GIGA_MODEL", "GigaChat"),
            "messages": _convert_messages_for_gigachat(messages),
            "temperature": temperature,
        }
        if tools:
            # GigaChat принимает functions (формат совместим с function-описанием из TOOL_SCHEMAS)
            payload["functions"] = [t["function"] for t in tools]
            payload["function_call"] = "auto"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        resp = requests.post(
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
            headers=headers,
            json=payload,
            verify=False,
            timeout=30,
        )
        if resp.status_code == 401:
            # Сброс протухшего токена и повтор
            self._token = None
            token = self.get_token()
            headers["Authorization"] = f"Bearer {token}"
            resp = requests.post(
                "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                headers=headers,
                json=payload,
                verify=False,
                timeout=30,
            )

        if resp.status_code != 200:
            raise RuntimeError(f"GigaChat API error {resp.status_code}: {resp.text}")

        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        parsed_calls: List[ToolCall] = []

        if "function_call" in message and message["function_call"]:
            fc = message["function_call"]
            raw_args = fc.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            parsed_calls.append(
                ToolCall(
                    id=message.get("functions_state_id") or str(uuid.uuid4()),
                    name=fc.get("name", ""),
                    arguments=raw_args if isinstance(raw_args, dict) else {},
                )
            )
        elif "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}
                parsed_calls.append(
                    ToolCall(
                        id=tc.get("id") or str(uuid.uuid4()),
                        name=fn.get("name", ""),
                        arguments=raw_args if isinstance(raw_args, dict) else {},
                    )
                )

        return LLMResponse(content=content, tool_calls=parsed_calls, raw=data)


_default_client: Optional[GigaChatClient] = None


def get_gigachat_client() -> GigaChatClient:
    global _default_client
    if _default_client is None:
        _default_client = GigaChatClient()
    return _default_client
