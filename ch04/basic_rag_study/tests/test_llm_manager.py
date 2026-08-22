"""LLM 호출과 프롬프트 조립 테스트."""

import pytest


class TestGenerateResponse:
    def test_fake_completion_requires_the_current_request_shape(
        self,
        fake_openai,
    ):
        with pytest.raises(TypeError):
            fake_openai.chat.completions.create(
                model="gpt-5.6-luna",
                messages=[],
                max_tokens=5,
            )

        with pytest.raises(TypeError):
            fake_openai.chat.completions.create(
                model="gpt-5.6-luna",
                messages=[],
                max_completion_tokens=5,
                unsupported_option=True,
            )

    def test_returns_the_message_content(
        self,
        llm_manager,
        fake_openai,
    ):
        fake_openai.completion_text = "안녕하세요"

        assert llm_manager.generate_response("질문") == "안녕하세요"

    def test_settings_supply_the_defaults(
        self,
        llm_manager,
        fake_openai,
        settings,
    ):
        llm_manager.generate_response("질문")

        call = fake_openai.completion_calls[0]
        assert call["model"] == settings.llm_model
        assert call["max_completion_tokens"] == settings.max_tokens
        assert call["reasoning_effort"] == settings.reasoning_effort
        assert "temperature" not in call

    def test_arguments_override_the_defaults(
        self,
        llm_manager,
        fake_openai,
    ):
        llm_manager.generate_response("질문", max_tokens=10, temperature=0.0)

        call = fake_openai.completion_calls[0]
        assert call["max_completion_tokens"] == 10
        assert "temperature" not in call

    def test_empty_content_becomes_an_empty_string(
        self,
        llm_manager,
        fake_openai,
    ):
        """SDK의 message.content는 Optional[str]이다.

        도구 호출만 하거나 길이 제한에 걸리면 None이 온다. 반환형이 str이라고
        해놓고 None을 흘려보내면 최종 응답에 'None'이 그대로 찍힌다.
        """
        fake_openai.completion_text = None

        assert llm_manager.generate_response("질문") == ""

    def test_api_failure_becomes_a_message_instead_of_an_exception(
        self,
        llm_manager,
    ):
        def boom(
            **_kwargs,
        ):
            raise RuntimeError("rate limited")

        llm_manager.client.chat.completions.create = boom

        result = llm_manager.generate_response("질문")

        # 대화가 끊기지 않도록 예외를 문자열로 바꿔 돌려준다
        assert "rate limited" in result


class TestPrompts:
    def test_planning_prompt_lists_every_action(
        self,
        llm_manager,
        planner,
    ):
        prompt = llm_manager.create_planning_prompt("질문", planner.available_actions)

        for action in planner.available_actions:
            assert action in prompt
        assert "질문" in prompt

    def test_rag_prompt_carries_query_and_context(
        self,
        llm_manager,
    ):
        prompt = llm_manager.create_rag_prompt("무엇인가", "문맥 본문")

        assert "무엇인가" in prompt
        assert "문맥 본문" in prompt

    def test_profile_prompt_carries_every_profile_field(
        self,
        llm_manager,
    ):
        prompt = llm_manager.create_profile_based_prompt(
            "질문",
            "문맥",
            {
                "expertise_level": "advanced",
                "background": "academic",
                "preferred_detail_level": "low",
            },
        )

        assert "advanced" in prompt
        assert "academic" in prompt
        assert "low" in prompt

    def test_profile_prompt_tolerates_a_partial_profile(
        self,
        llm_manager,
    ):
        prompt = llm_manager.create_profile_based_prompt("질문", "문맥", {})

        # 빠진 항목은 기본값으로 메운다
        assert "intermediate" in prompt

    @pytest.mark.parametrize("max_length", [50, 300])
    def test_summary_prompt_states_the_limit(
        self,
        llm_manager,
        max_length,
    ):
        prompt = llm_manager.create_summary_prompt("본문", max_length=max_length)

        assert str(max_length) in prompt
        assert "본문" in prompt

    def test_analysis_prompt_carries_query_and_context(
        self,
        llm_manager,
    ):
        prompt = llm_manager.create_analysis_prompt("질문", "문맥")

        assert "질문" in prompt
        assert "문맥" in prompt
