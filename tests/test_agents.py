"""基础测试 - 不需要 API key 的单元测试"""

import pytest
from src.tools.calculator import calculator
from src.tools.python_repl import python_repl
from src.tools.file_ops import read_file, write_file
from src.rag.splitter import split_documents
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


class TestFileOps:
    def test_write_and_read(self, tmp_path):
        file_path = str(tmp_path / "test.txt")
        write_result = write_file.invoke({"file_path": file_path, "content": "hello world"})
        assert "success" in write_result.lower()

        read_result = read_file.invoke({"file_path": file_path})
        assert read_result == "hello world"

    def test_read_nonexistent(self):
        result = read_file.invoke({"file_path": "/nonexistent/file.txt"})
        assert "not found" in result.lower()


class TestSplitter:
    def test_split(self):
        docs = [Document(page_content="a" * 2000)]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 550 for c in chunks)
