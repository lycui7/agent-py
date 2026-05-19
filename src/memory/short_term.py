"""短期记忆 — 管理当前会话的对话历史。"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger("agent")

Role = Literal["human", "ai"]


class ShortTermMemory:
    """In-memory chat history with configurable max turns.

    Each "turn" = one human message + one AI message (2 entries).
    When history exceeds max_turns, oldest turns are dropped.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: list[BaseMessage] = []

    def add_message(self, role: Role, content: str) -> None:
        """Add a message to history."""
        if role == "human":
            self._history.append(HumanMessage(content=content))
        else:
            self._history.append(AIMessage(content=content))
        self._trim()

    def get_history(self) -> list[BaseMessage]:
        """Return current chat history as LangChain message list."""
        return list(self._history)

    def get_tuples(self) -> list[tuple[str, str]]:
        """Return history as (role, content) tuples for prompt formatting."""
        result = []
        for msg in self._history:
            role = "human" if isinstance(msg, HumanMessage) else "ai"
            result.append((role, msg.content))
        return result

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()
        logger.debug("Short-term memory cleared")

    @property
    def length(self) -> int:
        return len(self._history)

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            dropped = len(self._history) - max_messages
            self._history = self._history[-max_messages:]
            logger.debug("Trimmed %d old messages from short-term memory", dropped)
