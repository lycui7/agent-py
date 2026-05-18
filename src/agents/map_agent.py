"""地图路线规划 Agent — 通过高德地图 MCP 服务提供地图工具。"""

import asyncio
from functools import wraps

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from src.agents.base import create_llm
from src.config import AMAP_API_KEY

_loop = None


def _get_loop():
    """Reuse a single event loop for all sync-wrapped async tool calls."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop


def _make_sync_tool(async_tool: StructuredTool) -> StructuredTool:
    """Wrap an async-only MCP tool to support sync invocation and catch errors."""
    async_func = async_tool.coroutine

    @wraps(async_func)
    def sync_wrapper(**kwargs):
        try:
            return _get_loop().run_until_complete(async_func(**kwargs))
        except Exception as e:
            return f"工具调用出错: {e}。请检查参数后重试。"

    return StructuredTool.from_function(
        func=sync_wrapper,
        coroutine=async_func,
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
    )


async def _get_mcp_tools():
    """Connect to Amap MCP server and get available tools."""
    client = MultiServerMCPClient(
        {
            "amap": {
                "url": f"https://mcp.amap.com/mcp?key={AMAP_API_KEY}",
                "transport": "streamable_http",
            }
        }
    )
    raw_tools = await client.get_tools()
    return [_make_sync_tool(t) for t in raw_tools]


def create_map_agent():
    """Create a map agent with Amap MCP tools.

    Tools include: route planning (driving/walking/cycling/transit),
    geocoding, POI search, weather query, etc.
    """
    tools = _get_loop().run_until_complete(_get_mcp_tools())
    llm = create_llm()

    system_prompt = """你是高德地图导航助手。你可以帮用户：
- 规划路线（驾车、步行、骑行、公交）
- 搜索地点和 POI
- 地理编码（地址转坐标）和逆地理编码
- 查询天气

请根据用户需求调用合适的工具。给出清晰、实用的出行建议。
注意：坐标格式为 "经度,纬度"，如 "116.397428,39.90923"。"""

    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)
