"""Settings가 환경을 언제, 어떻게 읽는지에 대한 테스트."""

import logging

import pytest
from pydantic import ValidationError

from config import DEFAULT_USER_PROFILE, Settings, load_mock_settings, load_settings


def _settings(
    **overrides,
) -> Settings:
    """환경을 타지 않는 Settings를 만든다."""
    return Settings(_env_file=None, openai_api_key="sk-test", **overrides)


class TestDefaults:
    def test_defaults_match_documented_values(self):
        s = _settings()

        assert s.llm_model == "gpt-5.6-luna"
        assert s.embedding_model == "text-embedding-3-small"
        assert s.knowledge_folder == "./knowledge_files"
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.max_tokens == 4096
        assert s.temperature == 0.7
        assert s.reasoning_effort == "none"
        assert s.log_level == "INFO"

    def test_default_user_profile_is_copied_per_instance(self):
        first = _settings()
        second = _settings()

        assert first.default_user_profile == DEFAULT_USER_PROFILE
        # 한 인스턴스를 건드려도 모듈 상수와 다른 인스턴스는 그대로여야 한다
        first.default_user_profile["expertise_level"] = "advanced"
        assert second.default_user_profile["expertise_level"] == "intermediate"
        assert DEFAULT_USER_PROFILE["expertise_level"] == "intermediate"

    def test_settings_are_frozen(self):
        s = _settings()

        with pytest.raises(ValidationError):
            s.chunk_size = 5


class TestEnvironmentReading:
    def test_environment_variables_are_read_and_converted(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("CHUNK_SIZE", "512")
        monkeypatch.setenv("TEMPERATURE", "0.1")

        s = Settings(_env_file=None)

        assert s.openai_api_key == "sk-from-env"
        # 문자열이 아니라 int/float으로 들어와야 한다
        assert s.chunk_size == 512
        assert isinstance(s.chunk_size, int)
        assert s.temperature == pytest.approx(0.1)

    def test_importing_config_does_not_read_the_environment(
        self,
        monkeypatch,
    ):
        """모듈 임포트만으로 환경을 읽으면 테스트가 환경에 묶인다."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        import importlib

        import config

        importlib.reload(config)  # 키가 없어도 임포트 자체는 성공해야 한다

        with pytest.raises(ValidationError):
            config.load_settings()

    def test_missing_api_key_is_reported(
        self,
        monkeypatch,
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)

        fields = [err["loc"][0] for err in exc_info.value.errors()]
        assert "openai_api_key" in fields

    def test_all_bad_fields_are_reported_at_once(
        self,
        monkeypatch,
    ):
        """하나씩 고쳐가며 재실행하지 않아도 되는 것이 pydantic을 쓴 이유다."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
        monkeypatch.setenv("TEMPERATURE", "99")

        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)

        fields = {err["loc"][0] for err in exc_info.value.errors()}
        assert fields == {"openai_api_key", "chunk_size", "temperature"}


class TestValidation:
    @pytest.mark.parametrize(
        ("chunk_size", "chunk_overlap"),
        [(100, 100), (100, 200)],
    )
    def test_overlap_must_be_smaller_than_chunk(
        self,
        chunk_size,
        chunk_overlap,
    ):
        """(chunk_size - chunk_overlap)이 0 이하면 _split_text가 끝나지 않는다."""
        with pytest.raises(ValidationError, match="chunk_size"):
            _settings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def test_overlap_smaller_than_chunk_is_accepted(self):
        assert _settings(chunk_size=100, chunk_overlap=99).chunk_overlap == 99

    @pytest.mark.parametrize("temperature", [-0.1, 2.1])
    def test_temperature_is_bounded(
        self,
        temperature,
    ):
        with pytest.raises(ValidationError):
            _settings(temperature=temperature)

    @pytest.mark.parametrize("chunk_size", [0, -1])
    def test_chunk_size_must_be_positive(
        self,
        chunk_size,
    ):
        with pytest.raises(ValidationError):
            _settings(chunk_size=chunk_size)


class TestLogLevel:
    @pytest.mark.parametrize(
        "given", ["debug", "Debug", "DEBUG"],
    )
    def test_lowercase_log_level_is_accepted(
        self,
        given,
    ):
        assert _settings(log_level=given).log_level == "DEBUG"

    def test_log_level_names_match_the_logging_module(self):
        for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            level = _settings(log_level=name).log_level
            assert isinstance(logging.getLevelName(level), int)

    def test_unknown_log_level_lists_the_valid_ones(self):
        with pytest.raises(ValidationError) as exc_info:
            _settings(log_level="verbose")

        message = str(exc_info.value)
        assert "DEBUG" in message
        assert "CRITICAL" in message


class TestMockSettings:
    def test_mock_settings_ignore_the_environment(
        self,
        monkeypatch,
    ):
        """생성자 인자가 환경변수보다 우선하므로 항상 같은 값이 나온다."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        monkeypatch.setenv("CHUNK_SIZE", "7")
        monkeypatch.setenv("LOG_LEVEL", "CRITICAL")

        mock = load_mock_settings()

        assert mock.openai_api_key == "sk-test-mock-key"
        assert mock.chunk_size == 1000
        assert mock.log_level == "INFO"

    def test_real_settings_do_read_the_environment(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        monkeypatch.setenv("CHUNK_SIZE", "512")

        assert load_settings().chunk_size == 512

    def test_mock_settings_cover_every_field(self):
        """필드를 추가하고 mock을 안 고치면 환경을 타게 되므로 여기서 잡는다."""
        explicit = load_mock_settings().model_fields_set

        assert set(Settings.model_fields) - explicit == {"default_user_profile"}
