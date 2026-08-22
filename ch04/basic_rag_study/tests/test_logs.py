"""로깅 설정 테스트. ch03/single_model_study/tests/test_logs.py를 참고했다."""

import logging
import sys

from logs import DEFAULT_LOG_LEVEL, LOG_FORMAT, _handler, configure_logging, get_logger


class TestHandler:
    def test_format_matches_ch03(self):
        assert LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert _handler.formatter._fmt == LOG_FORMAT

    def test_handler_targets_stdout(self):
        assert _handler.stream is sys.stdout

    def test_default_level_is_info(self):
        assert DEFAULT_LOG_LEVEL == logging.INFO


class TestGetLogger:
    def test_logger_keeps_the_module_name(self):
        """ch03과 달리 모듈 이름이 살아야 어느 단계에서 찍힌 줄인지 안다."""
        assert get_logger("rag_system").name == "rag_system"
        assert get_logger("agent").name == "agent"

    def test_repeated_calls_do_not_stack_handlers(self):
        first = get_logger("probe.repeat")
        before = len(first.handlers)

        second = get_logger("probe.repeat")

        assert second is first
        assert len(second.handlers) == before

    def test_logger_shares_the_single_handler(self):
        assert _handler in get_logger("probe.shared").handlers

    def test_logger_is_open_and_the_handler_filters(self):
        """레벨을 핸들러가 쥐고 있어야 configure_logging 한 번으로 바뀐다."""
        assert get_logger("probe.level").level == logging.DEBUG

    def test_records_reach_pytest(
        self,
        caplog,
    ):
        """propagate를 끄면 caplog가 레코드를 못 받는다."""
        log = get_logger("probe.caplog")

        with caplog.at_level(logging.INFO, logger="probe.caplog"):
            log.info("hello")

        assert any("hello" in record.message for record in caplog.records)

    def test_record_renders_with_the_module_name(self):
        log = get_logger("probe.render")
        record = log.makeRecord(
            log.name, logging.INFO, __file__, 0, "message body", None, None,
        )

        assert " - probe.render - INFO - message body" in _handler.formatter.format(
            record,
        )


class TestConfigureLogging:
    def test_level_applies_to_loggers_made_beforehand(self):
        log = get_logger("probe.before")
        original = _handler.level
        try:
            configure_logging(logging.ERROR)
            assert log.isEnabledFor(logging.INFO)  # 로거 자체는 열려 있고
            assert _handler.level == logging.ERROR  # 핸들러가 걸러낸다
        finally:
            _handler.setLevel(original)

    def test_accepts_a_level_name(self):
        original = _handler.level
        try:
            configure_logging("WARNING")
            assert _handler.level == logging.WARNING
        finally:
            _handler.setLevel(original)
