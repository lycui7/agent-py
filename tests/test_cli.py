"""CLI 命令解析测试 — 不启动 Agent，只测 handle_command 和路由逻辑"""

from unittest.mock import patch

import pytest

from src.main import handle_command, MODES


class TestHandleCommand:
    def test_quit_exits(self):
        with pytest.raises(SystemExit):
            handle_command("/quit")

    def test_switch_returns_true(self):
        assert handle_command("/switch") is True

    def test_clear_returns_false(self):
        assert handle_command("/clear") is False

    def test_normal_input_returns_false(self):
        assert handle_command("hello world") is False

    def test_switch_case_insensitive(self):
        assert handle_command("/SWITCH") is True

    def test_quit_case_insensitive(self):
        with pytest.raises(SystemExit):
            handle_command("/QUIT")


class TestModes:
    def test_four_modes_defined(self):
        assert len(MODES) == 4

    def test_mode_keys_are_strings(self):
        assert all(isinstance(k, str) for k in MODES.keys())

    def test_mode_values_are_tuples(self):
        for key, value in MODES.items():
            assert isinstance(value, tuple)
            assert len(value) == 2

    def test_mode_names(self):
        names = [v[0] for v in MODES.values()]
        assert "tool" in names
        assert "rag" in names
        assert "multi" in names
        assert "map" in names


class TestCheckApiKey:
    def test_missing_key_exits(self):
        from src.main import check_api_key

        with patch("src.main.OPENAI_API_KEY", ""):
            with pytest.raises(SystemExit):
                check_api_key()

    def test_placeholder_key_exits(self):
        from src.main import check_api_key

        with patch("src.main.OPENAI_API_KEY", "sk-your-key-here"):
            with pytest.raises(SystemExit):
                check_api_key()

    def test_valid_key_passes(self):
        from src.main import check_api_key

        with patch("src.main.OPENAI_API_KEY", "sk-real-key-123"):
            check_api_key()
