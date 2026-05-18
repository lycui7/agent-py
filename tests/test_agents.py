"""基础测试 - 不需要 API key 的单元测试"""

from unittest.mock import patch

from src.tools.calculator import calculator
from src.tools.python_repl import python_repl
from src.tools.file_ops import read_file, write_file
import src.tools.file_ops as file_ops_module
from src.rag.splitter import split_documents
from src.utils.retry import retry_with_backoff
from langchain_core.documents import Document


class TestCalculator:
    def test_basic_arithmetic(self):
        assert calculator.invoke({"expression": "2 + 3"}) == "5"

    def test_math_functions(self):
        result = calculator.invoke({"expression": "sqrt(144)"})
        assert result == "12.0"

    def test_invalid_expression(self):
        result = calculator.invoke({"expression": "invalid"})
        assert "error" in result.lower()


class TestPythonREPL:
    def test_print(self):
        result = python_repl.invoke({"code": "print('hello')"})
        assert "hello" in result

    def test_no_output(self):
        result = python_repl.invoke({"code": "x = 1 + 1"})
        assert "no output" in result.lower()

    def test_timeout_on_infinite_loop(self):
        result = python_repl.invoke({"code": "while True: pass"})
        assert "timed out" in result.lower()

    def test_timeout_on_long_sleep(self):
        result = python_repl.invoke({"code": "import time; time.sleep(30)"})
        assert "timed out" in result.lower()

    def test_error_shows_exception_type(self):
        result = python_repl.invoke({"code": "1 / 0"})
        assert "ZeroDivisionError" in result


class TestFileOps:
    def test_write_and_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_ops_module, "ALLOWED_ROOT", tmp_path)
        file_path = str(tmp_path / "test.txt")
        write_result = write_file.invoke(
            {"file_path": file_path, "content": "hello world"}
        )
        assert "success" in write_result.lower()

        read_result = read_file.invoke({"file_path": file_path})
        assert read_result == "hello world"

    def test_read_nonexistent_in_allowed_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_ops_module, "ALLOWED_ROOT", tmp_path)
        result = read_file.invoke({"file_path": str(tmp_path / "nonexistent.txt")})
        assert "not found" in result.lower()

    def test_read_outside_allowed_dir(self):
        result = read_file.invoke({"file_path": "/etc/passwd"})
        assert "access denied" in result.lower()

    def test_write_outside_allowed_dir(self):
        result = write_file.invoke({"file_path": "/tmp/hacked.txt", "content": "bad"})
        assert "access denied" in result.lower()


class TestSplitter:
    def test_split(self):
        docs = [Document(page_content="a" * 2000)]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 550 for c in chunks)


class TestRetry:
    def test_success_on_first_try(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    @patch("src.utils.retry.time.sleep")
    def test_retry_on_connection_error(self, mock_sleep):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network down")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3

    @patch("src.utils.retry.time.sleep")
    def test_fail_after_max_retries(self, mock_sleep):
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fail():
            raise TimeoutError("timeout")

        import pytest

        with pytest.raises(TimeoutError):
            always_fail()

    def test_no_retry_on_unexpected_exception(self):
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def value_error():
            raise ValueError("bad value")

        import pytest

        with pytest.raises(ValueError):
            value_error()
