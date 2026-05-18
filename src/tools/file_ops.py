from pathlib import Path
from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Read the content of a text file. Returns the file content as a string."""
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            return f"File not found: {file_path}"
        if path.stat().st_size > 100_000:
            return "File too large (>100KB). Please read a specific section."
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Read error: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist, overwrites if it does."""
    try:
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Write error: {e}"
