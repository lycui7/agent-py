"""LangChain Agent 学习项目 - CLI 入口 (流式输出版)"""

import asyncio
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live

from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage

from src.config import OPENAI_API_KEY, AMAP_API_KEY

console = Console()

MODES = {
    "1": ("tool", "工具调用 Agent (搜索/计算/代码/文件)"),
    "2": ("rag", "RAG 知识问答 (基于文档)"),
    "3": ("multi", "多 Agent 协作 (Supervisor 模式)"),
    "4": ("map", "地图路线规划 (高德 MCP)"),
}


def check_api_key():
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
        console.print("[red]Error: OPENAI_API_KEY not set.[/red]")
        console.print("Copy .env.example to .env and add your API key.")
        sys.exit(1)


def show_banner():
    banner = """# LangChain Agent 学习项目

**四种模式可选:**
1. 工具调用 Agent — 搜索、计算、Python 代码、文件操作
2. RAG 知识问答 — 加载文档后基于内容回答
3. 多 Agent 协作 — Supervisor 自动分配任务给 Worker
4. 地图路线规划 — 高德地图 MCP 服务

**命令:** `/switch` 切换模式 | `/load <file>` 加载文档(RAG) | `/clear` 清除记忆 | `/quit` 退出"""
    console.print(Markdown(banner))
    console.print()


def select_mode():
    console.print("[bold]选择模式:[/bold]")
    for key, (_, desc) in MODES.items():
        console.print(f"  {key}. {desc}")
    choice = Prompt.ask("输入编号", choices=["1", "2", "3", "4"], default="1")
    return MODES[choice][0]


# ─── 流式输出 Agent (模式 1) ─────────────────────────────────────────


def run_tool_agent():
    from src.agents.tool_agent import create_tool_agent

    agent = create_tool_agent()
    console.print(Panel("工具调用 Agent 已启动 (流式输出)", style="green"))
    console.print("输入问题，Agent 会使用工具来回答。\n")

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        if handle_command(user_input):
            return

        _stream_agent_response(
            agent=agent,
            input_data={"messages": [{"role": "user", "content": user_input}]},
            title="Agent",
            style="green",
        )


# ─── 流式输出 RAG (模式 2) ───────────────────────────────────────────


def run_rag_agent():
    from src.agents.rag_agent import RAGAgent

    rag = RAGAgent()
    console.print(Panel("RAG 知识问答已启动 (流式输出)", style="blue"))
    console.print("使用 /load <file> 加载文档，然后提问。\n")

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

        if user_input.startswith("/load "):
            file_path = user_input[6:].strip()
            try:
                chunks = rag.load_document(file_path)
                console.print(f"[green]Loaded {chunks} chunks from {file_path}[/green]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            continue

        if handle_command(user_input):
            return

        if not rag.chain:
            console.print("[red]No documents loaded. Use /load <file> first.[/red]")
            continue

        console.print("[dim]Searching documents and thinking...[/dim]")
        _stream_text(rag.stream_ask(user_input), title="RAG Agent", style="blue")


# ─── 流式输出多 Agent (模式 3) ───────────────────────────────────────


def run_multi_agent():
    from src.agents.multi_agent import create_multi_agent

    agent = create_multi_agent()
    console.print(Panel("多 Agent 协作已启动 (Supervisor 多轮调度 + 流式输出)", style="magenta"))
    console.print("提出复杂任务，Supervisor 会反复调度 Worker Agent 直到完成。\n")

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        if handle_command(user_input):
            return

        step = 0
        final_content = ""

        for event in agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode="updates",
        ):
            for node_name, output in event.items():
                if node_name == "supervisor":
                    step += 1
                    console.print(f"\n[bold yellow]━━ Step {step}: Supervisor deciding... ━━[/bold yellow]")
                    continue

                # Worker node
                label = "[research_agent]" if "research" in node_name else "[code_agent]"
                msgs = output.get("messages", [])
                if not msgs:
                    continue

                last_msg = msgs[-1]
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

                # 显示工具调用过程
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        console.print(f"  [dim]🔧 Calling tool: {tc['name']}({tc['args']})[/dim]")

                # 如果是工具返回结果
                if isinstance(last_msg, ToolMessage):
                    console.print(f"  [dim]📋 Tool result:[/dim] {content[:200]}")
                    continue

                # Agent 的最终回答 - 流式显示
                if content:
                    agent_label = "Research Agent" if "research" in node_name else "Code Agent"
                    console.print(f"\n[bold cyan]{agent_label} {label}:[/bold cyan]")
                    _stream_text(_iter_text(content), title=agent_label, style="dim")
                    final_content = content

        console.print("\n[bold yellow]━━ Task Complete ━━[/bold yellow]")
        if final_content:
            console.print(Panel(Markdown(final_content), title="Final Result", style="magenta"))


# ─── 地图路线规划 (模式 4) ────────────────────────────────────────────


def run_map_agent():
    if not AMAP_API_KEY or AMAP_API_KEY.startswith("your_"):
        console.print("[red]Error: AMAP_API_KEY not set.[/red]")
        console.print("Get a key at https://console.amap.com/dev/key, then add to .env")
        return

    from src.agents.map_agent import create_map_agent

    with console.status("[dim]Connecting to Amap MCP server...[/dim]"):
        agent = create_map_agent()

    console.print(Panel("地图路线规划已启动 (高德 MCP)", style="green"))
    console.print("示例: 从北京西站到天安门广场怎么走？ / 搜索附近的咖啡厅\n")

    while True:
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        if handle_command(user_input):
            return

        _stream_agent_response(
            agent=agent,
            input_data={"messages": [{"role": "user", "content": user_input}]},
            title="Map Agent",
            style="green",
        )


# ─── 流式输出工具函数 ─────────────────────────────────────────────────


def _process_stream_chunk(msg, full_text: str, tool_calls_shown: set) -> str:
    """Process a single streaming chunk and return updated text."""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            tc_id = tc.get("id", "")
            if tc_id not in tool_calls_shown:
                tool_calls_shown.add(tc_id)
                args_str = str(tc.get("args", ""))
                if len(args_str) > 150:
                    args_str = args_str[:150] + "..."
                full_text += f"\n[dim]🔧 Thinking: calling {tc['name']}({args_str})[/dim]\n"

    if isinstance(msg, ToolMessage):
        result_preview = msg.content[:200] if len(msg.content) > 200 else msg.content
        full_text += f"[dim]📋 Tool result: {result_preview}[/dim]\n\n"

    if isinstance(msg, AIMessageChunk) and msg.content:
        full_text += msg.content

    return full_text


def _stream_agent_response(agent, input_data: dict, title: str, style: str):
    """流式输出 LangGraph prebuilt agent 的响应。显示思考过程和最终答案。"""
    full_text = ""
    tool_calls_shown = set()

    with Live(console=console, refresh_per_second=10) as live:
        for chunk in agent.stream(input_data, stream_mode="messages"):
            msg, metadata = chunk
            full_text = _process_stream_chunk(msg, full_text, tool_calls_shown)
            if full_text:
                live.update(Panel(Markdown(full_text), title=title, style=style))

    console.print()


async def _async_stream_agent(agent, input_data: dict, title: str, style: str):
    """异步流式输出 Agent 响应（用于 MCP 工具）。"""
    full_text = ""
    tool_calls_shown = set()

    with Live(console=console, refresh_per_second=10) as live:
        async for chunk in agent.astream(input_data, stream_mode="messages"):
            msg, metadata = chunk
            full_text = _process_stream_chunk(msg, full_text, tool_calls_shown)
            if full_text:
                live.update(Panel(Markdown(full_text), title=title, style=style))

    console.print()


def _async_stream_agent_response(agent, input_data: dict, title: str, style: str):
    """同步包装器，调用异步流式输出。"""
    asyncio.run(_async_stream_agent(agent, input_data, title, style))


def _stream_text(text_iter, title: str, style: str):
    """流式逐字输出文本。"""
    full_text = ""
    with Live(console=console, refresh_per_second=10) as live:
        for token in text_iter:
            full_text += token
            live.update(Panel(Markdown(full_text), title=title, style=style))
    console.print()


def _iter_text(text: str):
    """将字符串拆成逐字迭代器，模拟流式效果。"""
    for char in text:
        yield char


# ─── 通用 ────────────────────────────────────────────────────────────


def handle_command(user_input: str) -> bool:
    """Handle special commands. Returns True if the caller should return."""
    cmd = user_input.strip().lower()
    if cmd == "/quit":
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)
    if cmd == "/switch":
        return True
    if cmd == "/clear":
        console.print("[green]Memory cleared.[/green]")
        return False
    return False


def main():
    check_api_key()
    show_banner()

    while True:
        mode = select_mode()
        console.print()

        if mode == "tool":
            run_tool_agent()
        elif mode == "rag":
            run_rag_agent()
        elif mode == "multi":
            run_multi_agent()
        elif mode == "map":
            run_map_agent()


if __name__ == "__main__":
    main()
