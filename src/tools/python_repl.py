import multiprocessing
from langchain_core.tools import tool

EXEC_TIMEOUT = 10  # seconds


def _run_code(code: str, result_queue: multiprocessing.Queue):
    import io
    import contextlib

    namespace = {}
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, namespace)
        result = output.getvalue()
        result_queue.put(
            ("ok", result if result else "Code executed successfully (no output).")
        )
    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output. Use this for data processing, calculations, or any Python task.

    The code runs in an isolated namespace. Print results to see them.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_run_code, args=(code, result_queue))
    process.start()
    process.join(timeout=EXEC_TIMEOUT)

    if process.is_alive():
        process.terminate()
        process.join()
        return f"Execution timed out after {EXEC_TIMEOUT}s. Code may contain an infinite loop."

    status, result = result_queue.get_nowait()
    return result
