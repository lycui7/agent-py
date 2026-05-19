"""长期记忆 — 基于 ChromaDB 持久化对话摘要。"""

import logging
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.rag.vectorstore import get_embeddings
from src.config import CHROMA_PERSIST_DIR

logger = logging.getLogger("agent")

COLLECTION_NAME = "memory"
MAX_SUMMARY_TOKENS = 500


class LongTermMemory:
    """Persistent conversation memory backed by ChromaDB.

    Stores conversation summaries as documents. On a new session,
    relevant past summaries can be retrieved by semantic similarity
    and injected into the prompt as context.
    """

    def __init__(self, collection_name: str = COLLECTION_NAME):
        self._collection_name = collection_name
        self._store: Chroma | None = None

    def _ensure_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=get_embeddings(),
                persist_directory=CHROMA_PERSIST_DIR,
            )
        return self._store

    def save_summary(self, summary: str, session_id: str | None = None) -> None:
        """Persist a conversation summary.

        Args:
            summary: The text summary of the conversation.
            session_id: Optional identifier for the session.
        """
        sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        doc = Document(
            page_content=summary,
            metadata={
                "session_id": sid,
                "timestamp": datetime.now().isoformat(),
                "type": "conversation_summary",
            },
        )
        store = self._ensure_store()
        store.add_documents([doc])
        logger.info("Saved conversation summary for session %s", sid)

    def search_summaries(self, query: str, k: int = 3) -> list[str]:
        """Search for relevant past summaries.

        Args:
            query: The search query (e.g., the user's current question).
            k: Number of results to return.

        Returns:
            List of summary texts, most relevant first.
        """
        store = self._ensure_store()
        try:
            results = store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.warning("Failed to search memory summaries: %s", e)
            return []

    def clear(self, session_id: str | None = None) -> None:
        """Clear stored summaries.

        Args:
            session_id: If provided, only clear that session's summaries.
                        If None, clear all.
        """
        store = self._ensure_store()
        try:
            if session_id:
                store.delete(where={"session_id": session_id})
                logger.info("Cleared memory for session %s", session_id)
            else:
                # Delete all documents in the collection
                collection = store._collection
                if collection.count() > 0:
                    collection.delete(where={})
                logger.info("Cleared all long-term memory")
        except Exception as e:
            logger.warning("Failed to clear memory: %s", e)
