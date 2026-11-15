"""
Инструменты агента с docstring-контрактами, Pydantic-валидацией и безопасной обёрткой run_tool.
"""
import ast
import json
import operator
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, ValidationError

# Опциональный коллбэк для подмены поиска в тестах без загрузки BERT/FAISS
_custom_search_fn: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None


def set_search_fn(fn: Optional[Callable[[str, int], List[Dict[str, Any]]]]) -> None:
    """Позволяет подменить функцию поиска (удобно для юнит-тестов)."""
    global _custom_search_fn
    _custom_search_fn = fn


def tool_search(query: str) -> str:
    """Поиск релевантных фрагментов в базе англоязычных бизнес-новостей (RAG / FAISS).
    Используй, когда нужен факт, событие, цифра или цитата из корпуса документов.
    Запрос (query) лучше формулировать на английском языке, например 'central bank interest rates'.
    Не используй для арифметических вычислений."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Пустой поисковый запрос")

    if _custom_search_fn is not None:
        chunks = _custom_search_fn(query, 3)
    else:
        from src.rag import get_rag
        rag = get_rag()
        chunks = rag.search(query, top_k=3)

    if not chunks:
        return "В базе документов релевантных фрагментов не найдено."

    lines = []
    for i, ch in enumerate(chunks, 1):
        cat = ch.get("category", "unknown")
        dist = ch.get("distance", 0.0)
        text = str(ch.get("text", "")).strip()
        lines.append(f"[{i}] (category={cat}, dist={dist:.3f}) {text}")
    return "\n".join(lines)


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Слишком большая степень")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError(f"Недопустимая операция в выражении: {type(node).__name__}")


def tool_calc(expression: str) -> str:
    """Вычислить арифметическое выражение, например '2*(3+4)' или '150 / 3'.
    Используй только для чисел и операторов +, -, *, /, //, %, ** и скобок."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Пустое арифметическое выражение")

    expr = expression.strip().replace(",", ".")
    tree = ast.parse(expr, mode="eval")
    result = _safe_eval_node(tree)
    if result.is_integer():
        return str(int(result))
    return str(round(result, 6))


def tool_classify_topic(text: str) -> str:
    """Определить бизнес-категорию текста (например: stock_market, government_policy, corporate_earnings).
    Используй, когда нужно классифицировать тему новости или фрагмента текста."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Пустой текст для классификации")

    from src.classifier import get_classifier
    clf = get_classifier()
    res = clf.predict(text)
    return json.dumps(
        {"category": res["category"], "confidence": res["confidence"]},
        ensure_ascii=False,
    )


class SearchArgs(BaseModel):
    query: str = Field(..., description="Поисковый запрос к базе англоязычных бизнес-новостей (на английском языке)")


class CalcArgs(BaseModel):
    expression: str = Field(..., description="Арифметическое выражение, например '2*(3+4)'")


class ClassifyTopicArgs(BaseModel):
    text: str = Field(..., description="Текст документа или новости для классификации темы")


TOOLS: Dict[str, Callable[..., str]] = {
    "search": tool_search,
    "calc": tool_calc,
    "classify_topic": tool_classify_topic,
}

TOOL_ARGS_MODELS: Dict[str, Type[BaseModel]] = {
    "search": SearchArgs,
    "calc": CalcArgs,
    "classify_topic": ClassifyTopicArgs,
}

# Единственный строковый параметр для каждого инструмента (используется в JSON-подходе)
_PRIMARY_ARG_FIELD: Dict[str, str] = {
    "search": "query",
    "calc": "expression",
    "classify_topic": "text",
}


def build_tool_schemas() -> List[Dict[str, Any]]:
    """Генерирует JSON Schema (OpenAI / GigaChat format) из Pydantic-моделей и docstring."""
    schemas: List[Dict[str, Any]] = []
    for name, fn in TOOLS.items():
        model = TOOL_ARGS_MODELS[name]
        params = model.model_json_schema()
        params.pop("title", None)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (fn.__doc__ or "").strip(),
                    "parameters": params,
                },
            }
        )
    return schemas


TOOL_SCHEMAS: List[Dict[str, Any]] = build_tool_schemas()


def run_tool(name: str, args: Union[Dict[str, Any], str]) -> str:
    """Безопасный вызов инструмента с валидацией аргументов и перехватом ошибок."""
    if name not in TOOLS:
        return f"Ошибка: неизвестный инструмент '{name}'. Доступны: {list(TOOLS)}."

    model_cls = TOOL_ARGS_MODELS[name]
    try:
        if isinstance(args, str):
            field_name = _PRIMARY_ARG_FIELD[name]
            validated = model_cls(**{field_name: args})
        elif isinstance(args, dict):
            validated = model_cls(**args)
        else:
            return f"Ошибка аргументов инструмента '{name}': ожидался dict или str."
        return str(TOOLS[name](**validated.model_dump()))
    except ValidationError as e:
        return f"Ошибка валидации аргументов инструмента {name}: {e}"
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"
