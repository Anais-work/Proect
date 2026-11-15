"""
Вариант 3 (Фреймворк — LangGraph):
Граф состояния агента с узлами `agent` и `tools` на базе `langgraph.graph.StateGraph`,
использующий те же инструменты (search, calc, classify_topic) и лимит итераций.
"""
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, TypedDict
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agents.llm_client import LLMResponse, get_gigachat_client
from src.agents.native_agent import SYSTEM_PROMPT
from src.agents.tools import TOOL_SCHEMAS, run_tool


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: int


def _lc_messages_to_dicts(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            converted.append({"role": "system", "content": str(msg.content)})
        elif isinstance(msg, HumanMessage):
            converted.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            item: Dict[str, Any] = {"role": "assistant", "content": str(msg.content or "")}
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.get("id") or str(uuid.uuid4()),
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["args"]},
                    }
                    for tc in msg.tool_calls
                ]
            converted.append(item)
        elif isinstance(msg, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name or "",
                    "content": str(msg.content),
                }
            )
    return converted


def _default_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
    return get_gigachat_client().chat(messages=messages, tools=tools)


def run_framework_agent(
    user_query: str,
    llm_fn: Optional[Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], LLMResponse]] = None,
    max_iterations: int = 5,
    trace: bool = True,
) -> Dict[str, Any]:
    """Запускает агента на базе LangGraph StateGraph и возвращает ответ + лог шагов."""
    llm = llm_fn or _default_llm
    steps: List[Dict[str, Any]] = []

    def agent_node(state: AgentState) -> Dict[str, Any]:
        dict_msgs = _lc_messages_to_dicts(state["messages"])
        resp = llm(dict_msgs, TOOL_SCHEMAS)
        tool_calls = [
            {"id": tc.id, "name": tc.name, "args": tc.arguments}
            for tc in resp.tool_calls
        ]
        ai_msg = AIMessage(content=resp.content or "", tool_calls=tool_calls)
        return {"messages": [ai_msg], "iterations": state["iterations"] + 1}

    def tools_node(state: AgentState) -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_messages: List[ToolMessage] = []
        step_idx = max(0, state["iterations"] - 1)
        if isinstance(last_msg, AIMessage):
            for tc in last_msg.tool_calls:
                result = run_tool(tc["name"], tc["args"])
                steps.append(
                    {
                        "step": step_idx,
                        "tool": tc["name"],
                        "args": tc["args"],
                        "result": result,
                    }
                )
                if trace:
                    print(f"[langgraph step {step_idx}] tool={tc['name']} args={tc['args']} -> {result[:120]}")
                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
        return {"messages": tool_messages}

    def route_after_agent(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            if state["iterations"] >= max_iterations:
                return "limit"
            return "tools"
        return END

    def limit_node(state: AgentState) -> Dict[str, Any]:
        return {
            "messages": [AIMessage(content="Не удалось получить ответ за отведенное число шагов.")]
        }

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("limit", limit_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "limit": "limit", END: END})
    graph.add_edge("tools", "agent")
    graph.add_edge("limit", END)

    app = graph.compile()
    final_state = app.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ],
            "iterations": 0,
        }
    )

    last_message = final_state["messages"][-1]
    return {
        "mode": "framework",
        "answer": str(last_message.content),
        "steps": steps,
        "iterations": final_state["iterations"],
    }
