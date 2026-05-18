from typing import Any, Iterator
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _handle_openai_bad_request, _handle_openai_api_error
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import ChatGenerationChunk

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, TEMPERATURE, MAX_TOKENS

# Global store: maps tool_call_id → reasoning_content from the previous response
_reasoning_store: dict[str, str] = {}


def _sanitize(text: str) -> str:
    """Remove surrogate characters that break JSON serialization."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


class ReasoningCaptureChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that captures and restores reasoning_content for mimo models.

    The mimo API requires reasoning_content to be passed back in multi-turn conversations.
    LangChain doesn't handle this, so we intercept at two points:

    1. _stream / _create_chat_result: capture reasoning_content from API responses.
       In streaming, raw chunks contain reasoning_content in choice.delta but
       _convert_delta_to_message_chunk doesn't extract it, so we override _stream
       to grab it from the raw chunk dict before conversion.
    2. _get_request_payload: after parent builds the payload, inject reasoning_content
       back into assistant messages with tool_calls that are missing it.
    """

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Override _stream to capture reasoning_content from raw streaming chunks."""
        import openai

        self._ensure_sync_client_available()
        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class = AIMessageChunk
        base_generation_info = {}

        # Accumulate reasoning and tool_call_ids across chunks
        accumulated_reasoning = ""
        accumulated_tool_call_ids: list[str] = []

        try:
            if self.include_response_headers:
                raw_response = self.client.with_raw_response.create(**payload)
                response = raw_response.parse()
                base_generation_info = {"headers": dict(raw_response.headers)}
            else:
                response = self.client.create(**payload)

            with response as response_stream:
                is_first_chunk = True
                for chunk in response_stream:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()

                    # Extract reasoning_content from raw chunk delta
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        rc = delta.get("reasoning_content")
                        if rc:
                            accumulated_reasoning += rc
                        # Collect tool_call ids
                        for tc in (delta.get("tool_calls") or []):
                            tc_id = tc.get("id")
                            if tc_id and tc_id not in accumulated_tool_call_ids:
                                accumulated_tool_call_ids.append(tc_id)

                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue
                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk

        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)

        # Store reasoning_content by tool_call_id for later injection
        if accumulated_reasoning and accumulated_tool_call_ids:
            sanitized = _sanitize(accumulated_reasoning)
            for tc_id in accumulated_tool_call_ids:
                _reasoning_store[tc_id] = sanitized
            # Clean up stale entries to prevent unbounded growth
            if len(_reasoning_store) > 100:
                keep_ids = set(accumulated_tool_call_ids)
                stale = [k for k in _reasoning_store if k not in keep_ids]
                for k in stale:
                    del _reasoning_store[k]

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "messages" in payload:
            for msg_dict in payload["messages"]:
                # Sanitize content (tool results from MCP may contain surrogates)
                content = msg_dict.get("content")
                if isinstance(content, str):
                    msg_dict["content"] = _sanitize(content)
                # Inject reasoning_content into assistant messages with tool_calls
                if msg_dict.get("role") == "assistant" and msg_dict.get("tool_calls"):
                    if "reasoning_content" not in msg_dict:
                        for tc in msg_dict.get("tool_calls", []):
                            tc_id = tc.get("id", "")
                            if tc_id in _reasoning_store:
                                msg_dict["reasoning_content"] = _reasoning_store[tc_id]
                                break
        return payload

    def _create_chat_result(self, response, generation_info=None):
        """Capture reasoning_content from non-streaming API responses."""
        result = super()._create_chat_result(response, generation_info)
        if hasattr(response, "choices") and response.choices:
            raw_msg = response.choices[0].message
            reasoning = getattr(raw_msg, "reasoning_content", None)
            if reasoning and result.generations:
                reasoning = _sanitize(reasoning)
                gen_msg = result.generations[0].message
                gen_msg.additional_kwargs["reasoning_content"] = reasoning
                if hasattr(gen_msg, "tool_calls") and gen_msg.tool_calls:
                    for tc in gen_msg.tool_calls:
                        _reasoning_store[tc["id"]] = reasoning
        return result


def create_llm(temperature: float = TEMPERATURE) -> ReasoningCaptureChatOpenAI:
    """Create a configured ChatOpenAI instance with mimo reasoning_content support."""
    return ReasoningCaptureChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
