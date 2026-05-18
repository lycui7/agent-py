from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_PERSIST_DIR, HF_TOKEN

EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        kwargs = {"model_name": EMBEDDING_MODEL}
        if HF_TOKEN:
            kwargs["model_kwargs"] = {"token": HF_TOKEN}
        _embeddings_instance = HuggingFaceEmbeddings(**kwargs)
    return _embeddings_instance


def create_vectorstore(documents: list[Document], collection_name: str = "default") -> Chroma:
    """Create a new Chroma vectorstore from documents."""
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def add_documents(vectorstore: Chroma, documents: list[Document]) -> int:
    """Add documents to an existing vectorstore. Returns number of new chunks."""
    vectorstore.add_documents(documents)
    return len(documents)


def get_vectorstore(collection_name: str = "default") -> Chroma:
    """Load an existing Chroma vectorstore."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
