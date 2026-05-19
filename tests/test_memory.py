"""Memory 模块测试 — 短期记忆 + 长期记忆，全部 mock，不需要 API key"""

from unittest.mock import MagicMock, patch

from src.memory.short_term import ShortTermMemory


# ─── ShortTermMemory ──────────────────────────────────────────────────


class TestShortTermMemory:
    def test_initial_state(self):
        mem = ShortTermMemory(max_turns=5)
        assert mem.get_history() == []
        assert mem.get_tuples() == []
        assert mem.length == 0

    def test_add_and_get_messages(self):
        mem = ShortTermMemory()
        mem.add_message("human", "hello")
        mem.add_message("ai", "hi there")

        history = mem.get_history()
        assert len(history) == 2
        assert history[0].content == "hello"
        assert history[1].content == "hi there"

    def test_get_tuples(self):
        mem = ShortTermMemory()
        mem.add_message("human", "q1")
        mem.add_message("ai", "a1")

        tuples = mem.get_tuples()
        assert tuples == [("human", "q1"), ("ai", "a1")]

    def test_trimming(self):
        mem = ShortTermMemory(max_turns=2)
        for i in range(5):
            mem.add_message("human", f"q{i}")
            mem.add_message("ai", f"a{i}")

        assert mem.length == 4  # 2 turns * 2 messages
        tuples = mem.get_tuples()
        assert tuples[0] == ("human", "q3")

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add_message("human", "q")
        mem.add_message("ai", "a")
        mem.clear()
        assert mem.length == 0
        assert mem.get_history() == []

    def test_get_history_returns_copy(self):
        mem = ShortTermMemory()
        mem.add_message("human", "q")
        history = mem.get_history()
        history.clear()
        assert mem.length == 1  # original unchanged


# ─── LongTermMemory ───────────────────────────────────────────────────


class TestLongTermMemory:
    @patch("src.memory.long_term.get_embeddings")
    @patch("src.memory.long_term.Chroma")
    def test_save_summary(self, mock_chroma_cls, mock_embed):
        from src.memory.long_term import LongTermMemory

        mock_store = MagicMock()
        mock_chroma_cls.return_value = mock_store

        mem = LongTermMemory()
        mem.save_summary("User asked about Python", session_id="s1")

        mock_store.add_documents.assert_called_once()
        doc = mock_store.add_documents.call_args[0][0][0]
        assert doc.page_content == "User asked about Python"
        assert doc.metadata["session_id"] == "s1"
        assert doc.metadata["type"] == "conversation_summary"

    @patch("src.memory.long_term.get_embeddings")
    @patch("src.memory.long_term.Chroma")
    def test_search_summaries(self, mock_chroma_cls, mock_embed):
        from src.memory.long_term import LongTermMemory
        from langchain_core.documents import Document

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="summary 1"),
            Document(page_content="summary 2"),
        ]
        mock_chroma_cls.return_value = mock_store

        mem = LongTermMemory()
        results = mem.search_summaries("Python question", k=2)

        assert results == ["summary 1", "summary 2"]
        mock_store.similarity_search.assert_called_once_with("Python question", k=2)

    @patch("src.memory.long_term.get_embeddings")
    @patch("src.memory.long_term.Chroma")
    def test_search_summaries_on_error(self, mock_chroma_cls, mock_embed):
        from src.memory.long_term import LongTermMemory

        mock_store = MagicMock()
        mock_store.similarity_search.side_effect = RuntimeError("db error")
        mock_chroma_cls.return_value = mock_store

        mem = LongTermMemory()
        results = mem.search_summaries("query")
        assert results == []

    @patch("src.memory.long_term.get_embeddings")
    @patch("src.memory.long_term.Chroma")
    def test_clear_all(self, mock_chroma_cls, mock_embed):
        from src.memory.long_term import LongTermMemory

        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_store._collection = mock_collection
        mock_chroma_cls.return_value = mock_store

        mem = LongTermMemory()
        mem.clear()

        mock_collection.delete.assert_called_once_with(where={})

    @patch("src.memory.long_term.get_embeddings")
    @patch("src.memory.long_term.Chroma")
    def test_clear_by_session(self, mock_chroma_cls, mock_embed):
        from src.memory.long_term import LongTermMemory

        mock_store = MagicMock()
        mock_chroma_cls.return_value = mock_store

        mem = LongTermMemory()
        mem.clear(session_id="s1")

        mock_store.delete.assert_called_once_with(where={"session_id": "s1"})
