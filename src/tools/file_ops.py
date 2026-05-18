import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger("agent")

# Only allow access to project directory and subdirectories
ALLOWED_ROOT = Path(__file__).resolve().parent.parent.parent


def _check_path(file_path: str) -> Path:
    path = Path(file_path).resolve()
    if not str(path).startswith(str(ALLOWED_ROOT)):
        raise PermissionError(
            f"Access denied: {file_path} is outside the allowed project directory"
        )
    return path


@tool
def read_file(file_path: str) -> str:
    """Read the content of a text file. Returns the file content as a string."""
    try:
        path = _check_path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        if path.stat().st_size > 100_000:
            return "File too large (>100KB). Please read a specific section."
        return path.read_text(encoding="utf-8")
    except PermissionError as e:
        logger.warning("read_file denied: %s", e)
        return str(e)
    except (OSError, UnicodeDecodeError) as e:
        logger.error("read_file error for %r: %s", file_path, e, exc_info=True)
        return f"Read error: {type(e).__name__}: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist, overwrites if it does."""
    try:
        path = _check_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {file_path}"
    except PermissionError as e:
        logger.warning("write_file denied: %s", e)
        return str(e)
    except OSError as e:
        logger.error("write_file error for %r: %s", file_path, e, exc_info=True)
        return f"Write error: {type(e).__name__}: {e}"
