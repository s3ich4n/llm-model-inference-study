import logging
import sys

from logs import logger


def test_logger_name():
    assert logger.name == "logs"


def test_logger_level_is_debug():
    assert logger.level == logging.DEBUG


def test_logger_has_stream_handler():
    stream_handlers = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    assert len(stream_handlers) == 1


def test_stream_handler_level_is_debug():
    handler = logger.handlers[0]
    assert handler.level == logging.DEBUG


def test_stream_handler_targets_stdout():
    handler = logger.handlers[0]
    assert handler.stream is sys.stdout


def test_formatter_format_string():
    handler = logger.handlers[0]
    assert handler.formatter._fmt == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def test_logger_emits_expected_format():
    # logs.py binds the handler to the sys.stdout object at import time, so
    # neither capsys nor capfd reliably observe writes to it in-process;
    # instead, render a record through the configured formatter directly.
    record = logger.makeRecord(
        logger.name, logging.DEBUG, __file__, 0, "hello from test", None, None
    )
    formatted = logger.handlers[0].formatter.format(record)
    assert " - logs - DEBUG - hello from test" in formatted


def test_model_executor_logs_on_init(caplog):
    from llm.model_executor import ModelExecutor

    with caplog.at_level(logging.DEBUG, logger="logs"):
        ModelExecutor()

    assert any(
        "Model executor initialized" in record.message for record in caplog.records
    )
