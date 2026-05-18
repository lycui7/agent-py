from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document


def load_file(file_path: str) -> list[Document]:
    """Load a single file. Supports .txt, .md, .pdf."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return loader.load()


def load_directory(dir_path: str, glob: str = "**/*") -> list[Document]:
    """Load all supported files from a directory."""
    loader = DirectoryLoader(
        dir_path,
        glob=glob,
        loader_cls=TextLoader,
        show_progress=True,
    )
    return loader.load()
