import re
from typing import Literal
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import create_react_agent

from src.agents.base import create_llm
from src.tools.search import web_search
from src.tools.calculator import calculator
from src.tools.python_repl import python_repl

_supervisor_llm = None


def _make_worker(tools, prompt):
    """Factory for worker agent nodes. Returns a function compatible with LangGraph."""
    cached_llm = None

    def worker(state: MessagesState):
        nonlocal cached_llm
        if cached_llm is None:
            cached_llm = create_llm()
        agent = create_react_agent(model=cached_llm, tools=tools, prompt=prompt)
        result = agent.invoke(state)
        return {"messages": result["messages"]}

    return worker


_research_agent = _make_worker(
    tools=[web_search],
    prompt="You are a research agent. Search the web to find information. Provide clear, factual results. Be thorough but concise.",
)

_code_agent = _make_worker(
    tools=[calculator, python_repl],
    prompt="You are a code agent. Write and execute Python code. Show results clearly. Be thorough but concise.",
)


def _supervisor(
    state: MessagesState,
) -> Literal["research_agent", "code_agent", "__end__"]:
    """Supervisor: reads the full conversation and decides the next step."""
    global _supervisor_llm
    if _supervisor_llm is None:
        _supervisor_llm = create_llm()

    messages = state["messages"]
    conversation = "\n".join(
        f"[{m.type}]: {m.content[:300]}" for m in messages if hasattr(m, "type")
    )

    response = _supervisor_llm.invoke(
        [
            {
                "role": "system",
                "content": """You are a supervisor managing a team of agents:
- research_agent: searches the web for information
- code_agent: writes and executes code, does calculations

Review the conversation so far. Decide what to do next:
- If the original task requires research AND it hasn't been done yet → respond "research_agent"
- If the original task requires code/calculation AND it hasn't been done yet → respond "code_agent"
- If all parts of the task are complete → respond "FINISH"

Be smart: if both research and code are needed, dispatch them in the right order.
If a worker has already provided useful results, don't re-dispatch it unless needed.

Respond with ONLY one word: research_agent, code_agent, or FINISH""",
            },
            {
                "role": "user",
                "content": f"Original task: {messages[0].content}\n\nConversation so far:\n{conversation}",
            },
        ]
    )

    decision = response.content.strip().lower()
    # Use word-boundary match to avoid false positives like "I finished the research"
    if re.search(r"\bfinish\b", decision):
        return "__end__"
    elif re.search(r"\bresearch\b", decision):
        return "research_agent"
    else:
        return "code_agent"


def create_multi_agent():
    """Create a multi-round multi-agent system with supervisor pattern.

    Architecture:
    ┌─────────────┐
    │  Supervisor  │──→ END (task complete)
    └──────┬───────┘
           │
    ┌──────┴──────┐
    ▼             ▼
    Research    Code       ← worker agents
    Agent       Agent
    │             │
    └──────┬──────┘
           │
       back to Supervisor  ← multi-round loop

    The supervisor reviews progress after each worker and decides:
    - dispatch another worker (multi-round)
    - finish if task is complete
    """
    graph = StateGraph(MessagesState)

    graph.add_node("supervisor", lambda state: state)  # pass-through node
    graph.add_node("research_agent", _research_agent)
    graph.add_node("code_agent", _code_agent)

    # START → supervisor
    graph.add_edge(START, "supervisor")

    # supervisor → worker or END
    graph.add_conditional_edges("supervisor", _supervisor)

    # workers → supervisor (loop back)
    graph.add_edge("research_agent", "supervisor")
    graph.add_edge("code_agent", "supervisor")

    return graph.compile()
