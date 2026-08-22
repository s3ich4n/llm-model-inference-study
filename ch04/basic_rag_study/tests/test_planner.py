"""계획 수립 로직 테스트. LLM 응답은 가짜 클라이언트가 준다."""

import json

import pytest


class TestFallbackPlan:
    """LLM 호출이 실패했을 때 쓰는 규칙 기반 계획."""

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("What is paging?", ["query_rag_with_context"]),
            ("How does it work?", ["query_rag_with_context"]),
            ("Explain tries", ["query_rag_with_context"]),
            ("Summarize the paper", ["query_rag_with_context", "generate_summary"]),
            ("Give me a brief", ["query_rag_with_context", "generate_summary"]),
            ("Analyze the tradeoffs", ["query_rag_with_context", "generate_analysis"]),
            ("Compare A and B", ["query_rag_with_context", "generate_analysis"]),
        ],
    )
    def test_keyword_routes_to_the_matching_plan(
        self,
        planner,
        query,
        expected,
    ):
        assert planner._create_fallback_plan(query)["plan"] == expected

    def test_unmatched_query_falls_back_to_the_general_plan(
        self,
        planner,
    ):
        plan = planner._create_fallback_plan("paging")

        assert plan["plan"] == [
            "query_rag_with_context",
            "generate_profile_based_response",
        ]

    def test_matching_is_case_insensitive(
        self,
        planner,
    ):
        assert planner._create_fallback_plan("SUMMARIZE this")["plan"] == (
            planner._create_fallback_plan("summarize this")["plan"]
        )

    def test_fallback_plans_are_always_valid(
        self,
        planner,
    ):
        for query in ["What is x?", "Summarize x", "Analyze x", "x"]:
            assert planner.validate_plan(planner._create_fallback_plan(query))


class TestParsePlanResponse:
    def test_extracts_json_wrapped_in_prose(
        self,
        planner,
    ):
        response = 'Sure! Here is the plan:\n{"plan": ["generate_summary"], ' \
                   '"reasoning": "because", "estimated_steps": 1}\nHope that helps.'

        assert planner._parse_plan_response(response)["plan"] == ["generate_summary"]

    def test_rejects_a_plan_missing_required_keys(
        self,
        planner,
    ):
        with pytest.raises(ValueError, match="required keys"):
            planner._parse_plan_response('{"plan": ["generate_summary"]}')

    def test_rejects_a_response_without_json(
        self,
        planner,
    ):
        with pytest.raises(ValueError, match="No JSON"):
            planner._parse_plan_response("I could not come up with a plan.")

    def test_rejects_malformed_json(
        self,
        planner,
    ):
        with pytest.raises(json.JSONDecodeError):
            planner._parse_plan_response('{"plan": [unquoted]}')


class TestValidatePlan:
    def test_accepts_a_plan_of_known_actions(
        self,
        planner,
    ):
        assert planner.validate_plan({"plan": ["generate_summary"]})

    @pytest.mark.parametrize(
        "plan",
        [
            {"plan": ["no_such_action"]},
            {"plan": "generate_summary"},
            {"reasoning": "no plan key"},
            "not a dict",
            None,
        ],
    )
    def test_rejects_anything_else(
        self,
        planner,
        plan,
    ):
        assert planner.validate_plan(plan) is False

    def test_every_available_action_passes_validation(
        self,
        planner,
    ):
        assert planner.validate_plan({"plan": planner.available_actions})


class TestGetActionSequence:
    def test_returns_the_actions_of_a_valid_plan(
        self,
        planner,
    ):
        plan = {"plan": ["generate_summary", "generate_analysis"]}

        assert planner.get_action_sequence(plan) == [
            "generate_summary",
            "generate_analysis",
        ]

    def test_returns_nothing_for_an_invalid_plan(
        self,
        planner,
    ):
        assert planner.get_action_sequence({"plan": ["bogus"]}) == []


class TestCreatePlan:
    def test_uses_the_llm_response_when_it_parses(
        self,
        container,
        fake_openai,
    ):
        fake_openai.completion_text = json.dumps(
            {"plan": ["generate_summary"], "reasoning": "ok", "estimated_steps": 1},
        )

        plan = container.planner().create_plan("Summarize the paper")

        assert plan["plan"] == ["generate_summary"]
        assert plan["reasoning"] == "ok"

    def test_falls_back_when_the_llm_returns_junk(
        self,
        container,
        fake_openai,
    ):
        fake_openai.completion_text = "죄송합니다, 계획을 세울 수 없습니다."

        plan = container.planner().create_plan("Summarize the paper")

        # 예외를 밖으로 던지지 않고 규칙 기반 계획으로 넘어간다
        assert plan["plan"] == ["query_rag_with_context", "generate_summary"]

    def test_planning_uses_a_low_temperature(
        self,
        container,
        fake_openai,
    ):
        container.planner().create_plan("What is paging?")

        assert fake_openai.completion_calls[0]["temperature"] == 0.3
