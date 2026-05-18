from langgraph.prebuilt import create_react_agent

from src.agents.base import create_llm
from src.tools.search import web_search
from src.tools.calculator import calculator
from src.tools.python_repl import python_repl
from src.tools.file_ops import read_file, write_file


def create_tool_agent():
    """Create a ReAct agent with general-purpose tools.

    The agent can search the web, do math, run Python code, and read/write files.
    It uses the ReAct pattern: Reason → Act → Observe → repeat.
    """
    llm = create_llm()
    tools = [web_search, calculator, python_repl, read_file, write_file]

    system_prompt = """You are a helpful assistant with access to these tools:
- web_search: search the internet for information
- calculator: evaluate math expressions
- python_repl: execute Python code
- read_file / write_file: interact with files

Think step by step. Use tools when needed to answer the user's question.
Always show your reasoning and the tool results."""

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )
