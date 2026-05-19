"""Tool Agent 测试 — mock LLM，不需要 API key"""

from unittest.mock import patch, MagicMock

from src.agents.tool_agent import create_tool_agent


class TestCreateToolAgent:
    @patch("src.agents.tool_agent.create_llm")
    def test_returns_agent(self, mock_create_llm):
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        agent = create_tool_agent()
        assert agent is not None
        assert hasattr(agent, "stream")

    @patch("src.agents.tool_agent.create_llm")
    def test_creates_llm(self, mock_create_llm):
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        create_tool_agent()
        mock_create_llm.assert_called_once()
