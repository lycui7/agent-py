# Agent-Py

基于 LangChain/LangGraph 的 AI Agent 学习项目，集成四种 Agent 模式，支持流式输出和工具调用。

## 功能特性

| 模式 | 说明 | 核心技术 |
|------|------|----------|
| **工具调用 Agent** | ReAct 模式，支持搜索、计算、Python 代码执行、文件操作 | LangGraph `create_react_agent` |
| **RAG 知识问答** | 加载文档（txt/md/pdf），基于内容进行多轮问答 | ChromaDB + HuggingFace Embeddings |
| **多 Agent 协作** | Supervisor 调度 Research / Code Worker，多轮循环完成复杂任务 | LangGraph StateGraph |
| **地图路线规划** | 路线规划、POI 搜索、地理编码、天气查询 | 高德地图 MCP |

## 快速开始

### 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
# 克隆项目
git clone https://github.com/lycui7/agent-py.git
cd agent-py

# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
```

编辑 `.env` 填入你的 API Key：

```env
OPENAI_API_KEY=sk-your-key-here          # 必填
OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1  # 兼容 OpenAI 协议的 API 地址
OPENAI_MODEL=mimo-v2.5-pro               # 模型名称
AMAP_API_KEY=your_amap_key_here          # 地图模式必填
HF_TOKEN=hf_your_token_here              # 可选，加速 Embedding 模型下载
```

### 启动

```bash
uv run python -m src.main
```

启动后选择模式编号（1-4）即可开始交互。

## 使用指南

### 模式 1：工具调用 Agent

直接输入问题，Agent 会自动选择合适的工具来回答：

```
You: 帮我算一下 (sqrt(144) + 3^2) * 2
You: 搜索一下 LangChain 最新版本
You: 用 Python 算斐波那契数列前 20 项
```

可用工具：`web_search` | `calculator` | `python_repl` | `read_file` | `write_file`

### 模式 2：RAG 知识问答

先加载文档，再基于文档内容提问：

```
You: /load docs/sample.md
You: 什么是 LangChain？
You: RAG 的工作流程是什么？
```

支持格式：`.txt` | `.md` | `.pdf`

### 模式 3：多 Agent 协作

输入复杂任务，Supervisor 自动分配给合适的 Worker：

```
You: 搜索 LangChain 的最新版本号，然后计算该版本号的平方根
```

Supervisor 会循环调度 Research Agent（搜索）和 Code Agent（计算）直到任务完成。

### 模式 4：地图路线规划

需要配置 `AMAP_API_KEY`（[申请地址](https://console.amap.com/dev/key)）：

```
You: 从北京西站到天安门广场怎么走？
You: 搜索附近的咖啡厅
You: 上海明天天气怎么样？
```

### 通用命令

| 命令 | 说明 |
|------|------|
| `/switch` | 切换模式 |
| `/load <file>` | 加载文档（仅 RAG 模式） |
| `/clear` | 清除对话记忆 |
| `/quit` | 退出程序 |

## 项目结构

```
src/
├── main.py              # CLI 入口，流式输出，模式路由
├── config.py            # 环境变量配置
├── agents/
│   ├── base.py          # 自定义 ChatOpenAI（处理 reasoning_content）
│   ├── tool_agent.py    # ReAct 工具调用 Agent
│   ├── rag_agent.py     # RAG 知识问答 Agent
│   ├── multi_agent.py   # Supervisor 多 Agent 协作
│   └── map_agent.py     # 高德地图 MCP Agent
├── tools/
│   ├── calculator.py    # 数学计算器
│   ├── search.py        # DuckDuckGo 网页搜索
│   ├── python_repl.py   # Python 代码沙盒执行
│   └── file_ops.py      # 文件读写（路径沙盒）
├── rag/
│   ├── loader.py        # 文档加载器（txt/md/pdf）
│   ├── splitter.py      # 文本分块
│   ├── vectorstore.py   # ChromaDB 向量存储
│   └── chain.py         # RAG Chain（LCEL）
├── memory/              # 对话记忆（开发中）
└── utils/
    └── retry.py         # 指数退避重试装饰器

tests/
├── test_core.py         # 核心类测试（LLM、Supervisor）
└── test_agents.py       # 工具和模块测试
```

## 开发

### 运行测试

```bash
uv run pytest -v
```

### 代码规范

项目使用 [ruff](https://docs.astral.sh/ruff/) 进行代码检查和格式化：

```bash
uvx ruff check src/ tests/
uvx ruff format --check src/ tests/

# 自动修复
uvx ruff check --fix src/ tests/
uvx ruff format src/ tests/
```

## 技术栈

- **Agent 框架**: LangChain + LangGraph
- **向量数据库**: ChromaDB
- **Embedding**: HuggingFace `shibing624/text2vec-base-chinese`
- **搜索**: DuckDuckGo
- **终端 UI**: Rich
- **包管理**: uv + hatchling
- **CI**: GitHub Actions（pytest + ruff）
