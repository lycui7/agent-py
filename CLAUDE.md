# Agent-Py

LangChain/LangGraph AI Agent 学习项目，四种 Agent 模式的 CLI 应用。

## 架构

```
src/main.py          -- CLI 入口 + 模式路由 + 流式输出
src/config.py        -- .env 配置加载
src/agents/          -- 四种 Agent 实现
  base.py            -- ReasoningCaptureChatOpenAI（适配 mimo 的 reasoning_content）
  tool_agent.py      -- ReAct 工具调用（LangGraph create_react_agent）
  rag_agent.py       -- RAG 问答（加载文档 -> 分块 -> 向量化 -> 检索 -> 生成）
  multi_agent.py     -- Supervisor + Research/Code Worker（LangGraph StateGraph）
  map_agent.py       -- 高德地图 MCP 集成
src/tools/           -- 工具实现（calculator, search, python_repl, file_ops）
src/rag/             -- RAG pipeline（loader, splitter, vectorstore, chain）
src/utils/           -- retry 指数退避装饰器
tests/               -- pytest 单测（不依赖外部 API）
```

## 技术栈

- Python 3.11+, LangChain 0.3+, LangGraph 0.2+
- ChromaDB (向量存储), HuggingFace text2vec-base-chinese (Embedding)
- Rich (终端 UI), DuckDuckGo (搜索), 高德 MCP (地图)
- uv (包管理), hatchling (构建), ruff (lint/format), pytest (测试)

## 代码规范

- **Linter/Formatter**: ruff（配置使用默认规则）
- 运行检查: `uvx ruff check src/ tests/` + `uvx ruff format --check src/ tests/`
- 自动修复: `uvx ruff check --fix src/ tests/` + `uvx ruff format src/ tests/`

## 测试

```bash
uv run pytest -v
```

- 33 个单测，全部 mock 外部 API，无需网络和 API Key
- `test_core.py` — LLM 类、Supervisor 逻辑
- `test_agents.py` — 工具函数、文本分块、重试机制
- CI 会在 push/PR to main 时自动运行 pytest + ruff

## 安全设计

- `python_repl`: multiprocessing 隔离 + 10s 超时
- `file_ops`: 路径沙盒限制在项目根目录内
- `calculator`: 受限 eval，仅允许 math 模块函数
- `search`: 指数退避重试（ConnectionError, TimeoutError, OSError）
