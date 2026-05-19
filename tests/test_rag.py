"""RAG pipeline 测试 — 全部 mock，不需要 API key 和外部服务"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag.loader import load_file
from src.rag.splitter import split_documents
from src.rag.chain import _format_docs, create_rag_chain


# ─── loader.py ────────────────────────────────────────────────────────


class TestLoadFile:
    def test_load_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        docs = load_file(str(f))
        assert len(docs) == 1
        assert "hello world" in docs[0].page_content

    def test_load_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nContent here", encoding="utf-8")
        docs = load_file(str(f))
        assert len(docs) == 1
        assert "Title" in docs[0].page_content

    def test_unsupported_type_raises(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(str(f))

    def test_nonexistent_file_raises(self):
        with pytest.raises(RuntimeError):
            load_file("/nonexistent/path.txt")


# ─── splitter.py ──────────────────────────────────────────────────────


class TestSplitterEdgeCases:
    def test_empty_document(self):
        docs = [Document(page_content="")]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 0

    def test_single_short_document(self):
        docs = [Document(page_content="short text")]
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0].page_content == "short text"

    def test_multiple_documents(self):
        docs = [
            Document(page_content="a" * 1500),
            Document(page_content="b" * 1500),
        ]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 4

    def test_preserves_metadata(self):
        docs = [Document(page_content="x" * 1000, metadata={"source": "test.md"})]
        chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
        assert all(c.metadata.get("source") == "test.md" for c in chunks)


# ─── chain.py ─────────────────────────────────────────────────────────


class TestFormatDocs:
    def test_format_single_doc(self):
        docs = [Document(page_content="hello")]
        assert _format_docs(docs) == "hello"

    def test_format_multiple_docs(self):
        docs = [Document(page_content="a"), Document(page_content="b")]
        assert _format_docs(docs) == "a\n\nb"

    def test_format_empty_list(self):
        assert _format_docs([]) == ""


class TestCreateRAGChain:
    @patch("src.rag.chain.create_llm")
    def test_returns_runnable(self, mock_create_llm):
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever

        chain = create_rag_chain(mock_vectorstore)
        assert chain is not None
        assert hasattr(chain, "invoke")
        mock_create_llm.assert_called_once_with(temperature=0)


# ─── vectorstore.py ───────────────────────────────────────────────────


class TestVectorstore:
    @patch("src.rag.vectorstore.Chroma")
    @patch("src.rag.vectorstore.get_embeddings")
    def test_create_vectorstore_calls_chroma(self, mock_embed, mock_chroma):
        from src.rag.vectorstore import create_vectorstore

        docs = [Document(page_content="test")]
        mock_chroma.from_documents.return_value = MagicMock()

        result = create_vectorstore(docs, collection_name="test_col")
        assert result is not None
        mock_chroma.from_documents.assert_called_once()
        call_kwargs = mock_chroma.from_documents.call_args
        assert call_kwargs.kwargs["collection_name"] == "test_col"

    @patch("src.rag.vectorstore.Chroma")
    @patch("src.rag.vectorstore.get_embeddings")
    def test_add_documents_returns_count(self, mock_embed, mock_chroma):
        from src.rag.vectorstore import add_documents

        mock_vs = MagicMock()
        docs = [Document(page_content="a"), Document(page_content="b")]
        count = add_documents(mock_vs, docs)
        assert count == 2
        mock_vs.add_documents.assert_called_once_with(docs)

    @patch("src.rag.vectorstore.Chroma")
    @patch("src.rag.vectorstore.get_embeddings")
    def test_get_vectorstore_creates_instance(self, mock_embed, mock_chroma):
        from src.rag.vectorstore import get_vectorstore

        result = get_vectorstore("my_col")
        assert result is not None
        mock_chroma.assert_called_once()
        call_kwargs = mock_chroma.call_args.kwargs
        assert call_kwargs["collection_name"] == "my_col"


# ─── RAGAgent ─────────────────────────────────────────────────────────


class TestRAGAgent:
    def _make_agent(self):
        from src.agents.rag_agent import RAGAgent

        return RAGAgent()

    def test_initial_state(self):
        agent = self._make_agent()
        assert agent.vectorstore is None
        assert agent.chain is None
        assert agent.chat_history == []

    def test_ask_without_docs_returns_message(self):
        agent = self._make_agent()
        result = agent.ask("hello")
        assert "No documents loaded" in result

    def test_stream_ask_without_docs_yields_message(self):
        agent = self._make_agent()
        tokens = list(agent.stream_ask("hello"))
        assert len(tokens) == 1
        assert "No documents loaded" in tokens[0]

    @patch("src.agents.rag_agent.create_rag_chain")
    @patch("src.agents.rag_agent.create_vectorstore")
    @patch("src.agents.rag_agent.split_documents")
    @patch("src.agents.rag_agent.load_file")
    def test_load_document_creates_chain(
        self, mock_load, mock_split, mock_create_vs, mock_create_chain
    ):
        mock_load.return_value = [Document(page_content="doc")]
        mock_split.return_value = [Document(page_content="chunk")]
        mock_vs = MagicMock()
        mock_create_vs.return_value = mock_vs
        mock_chain = MagicMock()
        mock_create_chain.return_value = mock_chain

        agent = self._make_agent()
        chunks = agent.load_document("test.md")

        assert chunks == 1
        assert agent.vectorstore is mock_vs
        assert agent.chain is mock_chain
        mock_create_chain.assert_called_once_with(mock_vs)

    @patch("src.agents.rag_agent.add_documents")
    @patch("src.agents.rag_agent.split_documents")
    @patch("src.agents.rag_agent.load_file")
    def test_load_document_appends_to_existing(self, mock_load, mock_split, mock_add):
        mock_load.return_value = [Document(page_content="doc")]
        mock_split.return_value = [Document(page_content="chunk")]

        agent = self._make_agent()
        agent.vectorstore = MagicMock()
        agent.chain = MagicMock()

        chunks = agent.load_document("another.md")
        assert chunks == 1
        mock_add.assert_called_once()

    def test_append_history_and_trimming(self):
        from src.agents.rag_agent import MAX_HISTORY_TURNS

        agent = self._make_agent()
        for i in range(MAX_HISTORY_TURNS + 5):
            agent._append_history(f"q{i}", f"a{i}")

        max_messages = MAX_HISTORY_TURNS * 2
        assert len(agent.chat_history) == max_messages
        assert agent.chat_history[0] == ("human", "q5")

    def test_clear_history(self):
        agent = self._make_agent()
        agent.chat_history = [("human", "q"), ("ai", "a")]
        agent.clear_history()
        assert agent.chat_history == []
