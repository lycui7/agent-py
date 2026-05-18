import io
import contextlib
from langchain_core.tools import tool


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output. Use this for data processing, calculations, or any Python task.

    The code runs in an isolated namespace. Print results to see them.
    """
    namespace = {}
    output = io.StringIO()

    try:
        with contextlib.redirect_stdout(output):
            exec(code, namespace)
        result = output.getvalue()
        return result if result else "Code executed successfully (no output)."
    except Exception as e:
        return f"Execution error: {e}"
