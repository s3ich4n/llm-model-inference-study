"""행동 실행기 테스트."""

import pytest


class TestDispatch:
    @pytest.mark.parametrize(
        "action",
        [
            "query_rag_with_context",
            "generate_profile_based_response",
            "generate_summary",
            "generate_analysis",
        ],
    )
    def test_every_known_action_runs(
        self,
        action_executor,
        action,
        fake_openai,
    ):
        result = action_executor.execute_action(action, "질문", context="문맥")

        assert result == fake_openai.completion_text

    def test_unknown_action_raises(
        self,
        action_executor,
    ):
        with pytest.raises(ValueError, match="Unknown action"):
            action_executor.execute_action("fly_to_the_moon", "질문")

    def test_planner_actions_and_executor_actions_agree(
        self,
        planner,
        action_executor,
    ):
        """계획에 나올 수 있는 행동은 전부 실행기가 알아야 한다."""
        for action in planner.available_actions:
            assert action_executor.get_action_description(action) != "Unknown action"

    def test_description_of_an_unknown_action(
        self,
        action_executor,
    ):
        assert action_executor.get_action_description("nope") == "Unknown action"


class TestContextHandling:
    def test_given_context_is_used_as_is(
        self,
        action_executor,
        fake_openai,
    ):
        action_executor.query_rag_with_context("질문", context="이미 있는 문맥")

        prompt = fake_openai.completion_calls[0]["messages"][0]["content"]
        assert "이미 있는 문맥" in prompt
        # 문맥을 받았으면 검색을 다시 하지 않는다
        assert fake_openai.embedding_calls == []

    def test_missing_context_is_fetched_from_the_rag_system(
        self,
        container,
        built_rag_system,
        fake_openai,
    ):
        container.action_executor().query_rag_with_context("paging")

        # 검색을 위해 질의를 임베딩했어야 한다
        assert fake_openai.embedding_calls == [["paging"]]

    def test_searching_without_a_built_database_raises(
        self,
        action_executor,
    ):
        with pytest.raises(ValueError, match="Vector database not built"):
            action_executor.query_rag_with_context("질문")


class TestProfile:
    def test_falls_back_to_the_settings_profile(
        self,
        action_executor,
        fake_openai,
    ):
        action_executor.generate_profile_based_response("질문", context="문맥")

        prompt = fake_openai.completion_calls[0]["messages"][0]["content"]
        assert "intermediate" in prompt
        assert "technical" in prompt

    def test_given_profile_wins(
        self,
        action_executor,
        fake_openai,
    ):
        action_executor.generate_profile_based_response(
            "질문",
            context="문맥",
            user_profile={
                "expertise_level": "beginner",
                "background": "business",
                "preferred_detail_level": "high",
            },
        )

        prompt = fake_openai.completion_calls[0]["messages"][0]["content"]
        assert "beginner" in prompt
        assert "business" in prompt
        assert "intermediate" not in prompt


class TestPrerequisites:
    def test_actions_are_blocked_until_the_database_is_built(
        self,
        action_executor,
    ):
        assert action_executor.validate_action_prerequisites(
            "query_rag_with_context",
        ) is False

    def test_actions_are_allowed_once_documents_exist(
        self,
        container,
        built_rag_system,
    ):
        executor = container.action_executor()

        assert executor.validate_action_prerequisites("query_rag_with_context")

    def test_unknown_action_never_passes(
        self,
        container,
        built_rag_system,
    ):
        assert container.action_executor().validate_action_prerequisites("nope") is False
