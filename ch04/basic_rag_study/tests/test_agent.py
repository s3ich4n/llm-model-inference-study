"""에이전트 오케스트레이션 테스트.

예전 이 파일은 `print` + `return True`로 이루어진 스크립트였다. assert가
하나도 없어 pytest가 무조건 통과로 집계했고, 실제로는 아무것도 검증하지
않았다. 지금은 조립, 계획 실행, 오류 처리를 실제로 확인한다.
"""

import json

import pytest


class TestWiring:
    def test_agent_uses_the_injected_components(
        self,
        agent,
        container,
    ):
        assert agent.settings is container.settings()
        assert agent.rag_system is container.rag_system()
        assert agent.action_executor is container.action_executor()

    def test_default_profile_comes_from_settings(
        self,
        agent,
        settings,
    ):
        assert agent.user_profile == settings.default_user_profile

    def test_given_profile_replaces_the_default(
        self,
        container,
    ):
        agent = container.agent(user_profile={"expertise_level": "beginner"})

        assert agent.user_profile == {"expertise_level": "beginner"}

    def test_profile_is_copied_not_shared(
        self,
        container,
    ):
        """에이전트가 프로필을 고치면 settings까지 바뀌어서는 안 된다."""
        agent = container.agent()

        agent.update_user_profile({"expertise_level": "advanced"})

        assert container.settings().default_user_profile["expertise_level"] == (
            "intermediate"
        )

    def test_get_user_profile_returns_a_copy(
        self,
        agent,
    ):
        returned = agent.get_user_profile()
        returned["expertise_level"] = "advanced"

        assert agent.user_profile["expertise_level"] == "intermediate"

    def test_update_merges_instead_of_replacing(
        self,
        agent,
    ):
        agent.update_user_profile({"expertise_level": "advanced"})

        assert agent.user_profile["expertise_level"] == "advanced"
        assert agent.user_profile["background"] == "technical"


class TestSystemStatus:
    def test_status_reports_an_empty_knowledge_base(
        self,
        agent,
    ):
        status = agent.get_system_status()

        assert status["rag_system"]["documents_loaded"] == 0
        assert status["rag_system"]["embeddings_available"] == 0

    def test_status_reflects_a_built_knowledge_base(
        self,
        container,
        built_rag_system,
    ):
        status = container.agent().get_system_status()

        assert status["rag_system"]["documents_loaded"] == 3

    def test_status_reports_the_configured_models(
        self,
        agent,
        settings,
    ):
        status = agent.get_system_status()

        assert status["llm_manager"]["model"] == settings.llm_model
        assert status["llm_manager"]["embedding_model"] == settings.embedding_model

    def test_status_lists_the_available_actions(
        self,
        agent,
        planner,
    ):
        assert agent.get_system_status()["available_actions"] == planner.available_actions


class TestBuildKnowledgeBase:
    def test_delegates_to_the_rag_system(
        self,
        agent,
        monkeypatch,
    ):
        calls = []
        monkeypatch.setattr(
            agent.rag_system,
            "build_vector_db",
            lambda force_rebuild=False: calls.append(force_rebuild),
        )

        agent.build_knowledge_base()
        agent.build_knowledge_base(force_rebuild=True)

        assert calls == [False, True]


class TestProcessQueryWithoutPlanning:
    def test_answers_directly(
        self,
        container,
        built_rag_system,
        fake_openai,
    ):
        fake_openai.completion_text = "답변 본문"

        result = container.agent().process_query("paging", use_planning=False)

        assert result["success"] is True
        assert result["plan"] is None
        assert result["final_response"] == "답변 본문"

    def test_unbuilt_knowledge_base_is_reported_as_a_failure(
        self,
        agent,
    ):
        result = agent.process_query("paging", use_planning=False)

        assert result["success"] is False
        assert "Vector database not built" in result["final_response"]
        assert "error" in result


class TestProcessQueryWithPlanning:
    def test_runs_the_planned_actions_in_order(
        self,
        container,
        built_rag_system,
        fake_openai,
    ):
        fake_openai.completion_text = json.dumps(
            {
                "plan": ["query_rag_with_context", "generate_summary"],
                "reasoning": "이유",
                "estimated_steps": 2,
            },
        )

        result = container.agent().process_query("paging")

        assert result["success"] is True
        assert result["plan"]["plan"] == [
            "query_rag_with_context",
            "generate_summary",
        ]
        assert len(result["results"]) == 2

    def test_actions_are_skipped_until_the_knowledge_base_exists(
        self,
        agent,
    ):
        """prerequisites를 통과 못 하면 행동을 건너뛰고 빈 결과가 나온다."""
        result = agent.process_query("paging")

        assert result["success"] is True
        assert result["results"] == []
        assert result["final_response"] == "No response generated"

    def test_a_failing_action_does_not_stop_the_rest(
        self,
        container,
        built_rag_system,
        fake_openai,
        monkeypatch,
    ):
        agent = container.agent()
        monkeypatch.setattr(
            agent.planner,
            "get_action_sequence",
            lambda _plan: ["generate_summary", "generate_analysis"],
        )
        calls = {"n": 0}

        def flaky(
            action_name,
            *args,
            **kwargs,
        ):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("첫 행동 실패")
            return "두 번째 행동 성공"

        monkeypatch.setattr(agent.action_executor, "execute_action", flaky)

        result = agent.process_query("paging")

        assert len(result["results"]) == 2
        assert "첫 행동 실패" in result["results"][0]
        assert result["final_response"] == "두 번째 행동 성공"

    def test_a_substantial_result_is_passed_on_as_context(
        self,
        container,
        built_rag_system,
        monkeypatch,
    ):
        agent = container.agent()
        monkeypatch.setattr(
            agent.planner,
            "get_action_sequence",
            lambda _plan: ["generate_summary", "generate_analysis"],
        )
        seen = []

        def record(
            action_name,
            query,
            context,
            user_profile,
        ):
            seen.append(context)
            return "가" * 60  # 50자를 넘겨야 다음 행동의 문맥이 된다

        monkeypatch.setattr(agent.action_executor, "execute_action", record)

        agent.process_query("paging")

        assert seen[0] == ""
        assert seen[1] == "가" * 60

    def test_a_short_result_is_not_passed_on(
        self,
        container,
        built_rag_system,
        monkeypatch,
    ):
        agent = container.agent()
        monkeypatch.setattr(
            agent.planner, "get_action_sequence", lambda _plan: ["generate_summary"] * 2,
        )
        seen = []

        def record(
            action_name,
            query,
            context,
            user_profile,
        ):
            seen.append(context)
            return "짧다"

        monkeypatch.setattr(agent.action_executor, "execute_action", record)

        agent.process_query("paging")

        assert seen == ["", ""]


class TestEmptyModelResponse:
    def test_none_content_does_not_leak_into_the_final_response(
        self,
        container,
        built_rag_system,
        fake_openai,
    ):
        """모델이 빈 응답을 줘도 사용자에게 None이 보이면 안 된다."""
        fake_openai.completion_text = None

        result = container.agent().process_query("paging", use_planning=False)

        assert result["success"] is True
        assert result["final_response"] == ""


class TestSearchKnowledgeBase:
    def test_passes_the_query_through(
        self,
        container,
        built_rag_system,
    ):
        results = container.agent().search_knowledge_base("paging", k=2)

        assert len(results) == 2

    def test_searching_before_building_raises(
        self,
        agent,
    ):
        with pytest.raises(ValueError, match="Vector database not built"):
            agent.search_knowledge_base("paging")
