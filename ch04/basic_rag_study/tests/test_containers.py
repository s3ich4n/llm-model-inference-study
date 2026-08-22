"""컨테이너가 무엇을 공유하고 무엇을 새로 만드는지에 대한 테스트."""

import pytest
from dependency_injector import providers
from pydantic import ValidationError

from config import Settings
from containers import Container


class TestSharing:
    def test_heavy_components_are_shared(
        self,
        container,
    ):
        """Singleton이므로 몇 번을 꺼내도 같은 객체여야 한다."""
        assert container.rag_system() is container.rag_system()
        assert container.llm_manager() is container.llm_manager()
        assert container.settings() is container.settings()

    def test_one_openai_client_serves_every_component(
        self,
        container,
    ):
        client = container.openai_client()

        assert container.rag_system().client is client
        assert container.llm_manager().client is client

    def test_agent_receives_the_shared_components(
        self,
        container,
    ):
        agent = container.agent()

        assert agent.rag_system is container.rag_system()
        assert agent.llm_manager is container.llm_manager()
        assert agent.planner is container.planner()
        assert agent.action_executor is container.action_executor()

    def test_agent_is_a_factory(
        self,
        container,
    ):
        """프로필이 다른 에이전트를 여러 개 만들 수 있어야 한다."""
        first = container.agent()
        second = container.agent(user_profile={"expertise_level": "beginner"})

        assert first is not second
        assert first.user_profile["expertise_level"] == "intermediate"
        assert second.user_profile["expertise_level"] == "beginner"
        # 그래도 무거운 구성 요소는 공유한다
        assert first.rag_system is second.rag_system


class TestOverriding:
    def test_settings_override_reaches_every_component(
        self,
        container,
    ):
        container.settings.override(
            providers.Singleton(
                lambda: Settings(
                    _env_file=None, openai_api_key="sk-x", chunk_size=42, chunk_overlap=1,
                ),
            ),
        )

        assert container.rag_system().settings.chunk_size == 42
        assert container.agent().settings.chunk_size == 42

    def test_environment_is_untouched_without_an_override(
        self,
        monkeypatch,
    ):
        """settings를 꺼내기 전에는 환경을 읽지 않는다."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        c = Container()  # 여기까지는 아무 일도 없어야 한다

        with pytest.raises(ValidationError):
            c.settings()


class TestLifecycle:
    def test_init_resources_applies_the_log_level(
        self,
        container,
    ):
        from logs import _handler

        container.settings.override(
            providers.Singleton(
                lambda: Settings(
                    _env_file=None, openai_api_key="sk-x", log_level="CRITICAL",
                ),
            ),
        )
        original = _handler.level
        try:
            container.init_resources()
            assert _handler.level == 50  # logging.CRITICAL
        finally:
            _handler.setLevel(original)
