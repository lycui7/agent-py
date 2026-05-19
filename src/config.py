import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI 兼容接口配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# HuggingFace 配置
HF_TOKEN = os.getenv("HF_TOKEN", "")

# 高德地图 API 配置
AMAP_API_KEY = os.getenv("AMAP_API_KEY", "")

# Chroma 配置
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# 日志级别 (DEBUG / INFO / WARNING / ERROR)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Agent 配置
MAX_TOKENS = 4096
TEMPERATURE = 0.7
