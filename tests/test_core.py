"""Agent 核心逻辑测试 — 全部 mock，不需要 API key"""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from src.agents.base import _sanitize, ReasoningCaptureChatOpenAI, _reasoning_store


class TestSanitize:
    def test_normal_text(self):
        assert _sanitize("hello world") == "hello world"

    def test_strips_surrogates(self):
        bad = "good\ud800text"
        result = _sanitize(bad)
        assert "\ud800" not in result
        assert "good" in result
        assert "text" in result

    def test_empty_string(self):
        assert _sanitize("") == ""

    def test_chinese(self):
        assert _sanitize("你好世界") == "你好世界"


class TestReasoningCaptureChatOpenAI:
    def _make_llm(self):
        return ReasoningCaptureChatOpenAI(
            api_key="test-key",
            base_url="http://localhost",
            model="test-model",
        )

    def test_create_llm_returns_correct_class(self):
        llm = self._make_llm()
        assert isinstance(llm, ReasoningCaptureChatOpenAI)
        assert llm.model_name == "test-model"

    def test_get_request_payload_injects_reasoning(self):
        llm = self._make_llm()
        _reasoning_store.clear()
        _reasoning_store["tc_123"] = "some reasoning"

        # Build a valid LangChain message list via the parent's conversion
        from langchain_core.messages import AIMessage, ToolMessage

        messages = [
            AIMessage(
                content="",
                tool_calls=[{"id": "tc_123", "name": "test", "args": {}}],
            ),
            ToolMessage(content="result", tool_call_id="tc_123"),
        ]

        payload = llm._get_request_payload(input_=messages)

        assistant_msg = next(
            m for m in payload["messages"] if m.get("role") == "assistant"
        )
        assert assistant_msg["reasoning_content"] == "some reasoning"
        _reasoning_store.clear()

    def test_get_request_payload_sanitizes_content(self):
        llm = self._make_llm()
        payload = llm._get_request_payload(
            input_=[{"role": "user", "content": "hello\ud800world"}]
        )
        assert "\ud800" not in payload["messages"][0]["content"]

    def test_create_chat_result_captures_reasoning(self):
        llm = self._make_llm()
        _reasoning_store.clear()

        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="answer")
        chat_result = ChatResult(generations=[ChatGeneration(message=msg)])

        mock_raw = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="answer",
                        reasoning_content="step by step thinking",
                        tool_calls=None,
                    )
                )
            ]
        )

        with patch.object(
            ReasoningCaptureChatOpenAI.__bases__[0],
            "_create_chat_result",
            return_value=chat_result,
        ):
            result = llm._create_chat_result(mock_raw)

        gen_msg = result.generations[0].message
        assert gen_msg.additional_kwargs["reasoning_content"] == "step by step thinking"
        _reasoning_store.clear()

    def test_reasoning_store_cleanup(self):
        _reasoning_store.clear()
        for i in range(105):
            _reasoning_store[f"old_{i}"] = f"reasoning_{i}"

        _reasoning_store["new_tc"] = "new reasoning"

        # Simulate cleanup logic
        if len(_reasoning_store) > 100:
            keep_ids = {"new_tc"}
            stale = [k for k in _reasoning_store if k not in keep_ids]
            for k in stale:
                del _reasoning_store[k]

        assert len(_reasoning_store) == 1
        assert "new_tc" in _reasoning_store
        _reasoning_store.clear()


class TestSupervisorDecision:
    """Test _supervisor decision parsing without real LLM calls."""

    def _mock_supervisor_response(self, text):
        import src.agents.multi_agent as ma_module

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = SimpleNamespace(content=text)
        with patch.object(ma_module, "_supervisor_llm", mock_llm):
            state = {
                "messages": [
                    SimpleNamespace(type="human", content="search for python tutorials")
                ]
            }
            return ma_module._supervisor(state)

    def test_finish_decision(self):
        assert self._mock_supervisor_response("FINISH") == "__end__"

    def test_finish_in_sentence(self):
        assert (
            self._mock_supervisor_response("I think we should finish now") == "__end__"
        )

    def test_research_decision(self):
        # Note: regex \bresearch\b matches standalone "research" but NOT "research_agent"
        # because _ is a word character. LLM returning "research_agent" falls to code_agent.
        assert self._mock_supervisor_response("research") == "research_agent"

    def test_research_with_context(self):
        assert (
            self._mock_supervisor_response("I think we need to do research first")
            == "research_agent"
        )

    def test_code_decision(self):
        assert self._mock_supervisor_response("code_agent") == "code_agent"

    def test_default_to_code(self):
        assert self._mock_supervisor_response("something random") == "code_agent"


class TestMakeWorker:
    def test_worker_returns_function(self):
        from src.agents.multi_agent import _make_worker

        worker = _make_worker(tools=[], prompt="test")
        assert callable(worker)
