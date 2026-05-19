"""日志模块测试 — 不依赖外部服务"""

import logging

from src.utils.logger import setup_logging


class TestSetupLogging:
    def _cleanup_logger(self):
        logger = logging.getLogger("agent")
        logger.handlers.clear()

    def test_creates_handlers(self):
        setup_logging(level="DEBUG", log_to_file=True)
        logger = logging.getLogger("agent")
        assert len(logger.handlers) == 2  # console + file
        self._cleanup_logger()

    def test_console_only(self):
        setup_logging(level="INFO", log_to_file=False)
        logger = logging.getLogger("agent")
        assert len(logger.handlers) == 1
        self._cleanup_logger()

    def test_log_level_applied(self):
        setup_logging(level="WARNING", log_to_file=False)
        logger = logging.getLogger("agent")
        assert logger.level == logging.WARNING
        self._cleanup_logger()

    def test_propagate_disabled(self):
        setup_logging(level="INFO", log_to_file=False)
        logger = logging.getLogger("agent")
        assert logger.propagate is False
        self._cleanup_logger()

    def test_default_level_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        # Re-import to pick up env change
        import importlib
        import src.utils.logger as logger_mod

        importlib.reload(logger_mod)
        logger_mod.setup_logging(log_to_file=False)
        logger = logging.getLogger("agent")
        assert logger.level == logging.ERROR
        self._cleanup_logger()
        # Restore module state
        importlib.reload(logger_mod)

    def test_file_handler_creates_log_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.logger.LOG_DIR", tmp_path / "logs")
        monkeypatch.setattr(
            "src.utils.logger.LOG_FILE", tmp_path / "logs" / "agent.log"
        )
        setup_logging(level="INFO", log_to_file=True)
        assert (tmp_path / "logs").exists()
        self._cleanup_logger()

    def test_log_writes_to_file(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        monkeypatch.setattr("src.utils.logger.LOG_DIR", log_dir)
        monkeypatch.setattr("src.utils.logger.LOG_FILE", log_dir / "agent.log")
        setup_logging(level="DEBUG", log_to_file=True)

        logger = logging.getLogger("agent")
        logger.info("test message 12345")

        log_content = (log_dir / "agent.log").read_text(encoding="utf-8")
        assert "test message 12345" in log_content
        self._cleanup_logger()
