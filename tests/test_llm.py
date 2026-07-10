import pytest

from mcp_dataframe_qa.llm import (
    LLMConfig,
    LLMConfigurationError,
    LLMPlanner,
    LLMResponseError,
    extract_json_object,
    resolve_llm_config,
)
from mcp_dataframe_qa.schemas import AnalysisPlan


def test_resolve_llm_config_detects_provider_from_key() -> None:
    config = resolve_llm_config(env={"ANTHROPIC_API_KEY": "test-key"})
    assert config.provider == "anthropic"
    assert config.api_key == "test-key"
    assert config.model


def test_resolve_llm_config_requires_key() -> None:
    try:
        resolve_llm_config(provider="openai", env={})
    except LLMConfigurationError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing-key configuration error.")


def test_extract_json_object_from_fenced_response() -> None:
    payload = extract_json_object(
        """
        ```json
        {"filters": [], "group_by": [], "metrics": [{"fn": "count", "column": "*", "as": "count"}]}
        ```
        """
    )
    assert payload["metrics"][0]["fn"] == "count"


def test_llm_planner_validates_provider_json(monkeypatch) -> None:
    def fake_complete(self: LLMPlanner, system: str, user: str) -> str:
        assert "AnalysisPlan" in system
        assert "median_list_price" in user
        return """
        {
          "filters": [],
          "group_by": ["region_name"],
          "metrics": [{"fn": "median", "column": "median_list_price", "as": "median_list_price"}],
          "sort": [{"column": "median_list_price", "direction": "desc"}],
          "limit": 10
        }
        """

    monkeypatch.setattr(LLMPlanner, "complete", fake_complete)
    planner = LLMPlanner(LLMConfig(provider="openai", api_key="test-key", model="test-model"))
    plan = planner.plan(
        "What are the top metros by median list price?",
        {
            "dataset_id": "default",
            "table_name": "zillow_metro_market",
            "row_count": 91872,
            "columns": {
                "region_name": {"dtype": "str", "semantic_type": "dimension"},
                "median_list_price": {"dtype": "float64", "semantic_type": "currency"},
            },
        },
    )
    assert isinstance(plan, AnalysisPlan)
    assert plan.group_by == ["region_name"]
    assert plan.metrics[0].column == "median_list_price"


def test_llm_planner_retries_once_after_invalid_schema(monkeypatch) -> None:
    responses = [
        # Wrong field name: "literal" instead of "value", a mistake seen in the wild
        # since the op itself is called "literal".
        """
        {
          "filters": [],
          "group_by": [],
          "metrics": [{"fn": "count", "column": "*", "as": "count"}],
          "derive": [{"name": "bad", "expr": {"op": "literal", "literal": 1}}]
        }
        """,
        """
        {
          "filters": [],
          "group_by": [],
          "metrics": [{"fn": "count", "column": "*", "as": "count"}]
        }
        """,
    ]
    calls: list[str] = []

    def fake_complete(self: LLMPlanner, system: str, user: str) -> str:
        calls.append(user)
        return responses[len(calls) - 1]

    monkeypatch.setattr(LLMPlanner, "complete", fake_complete)
    planner = LLMPlanner(LLMConfig(provider="openai", api_key="test-key", model="test-model"))
    plan = planner.plan("How many rows are there?", {"columns": {}})

    assert isinstance(plan, AnalysisPlan)
    assert plan.metrics[0].fn == "count"
    assert len(calls) == 2
    # The retry prompt carries the concrete validation error forward, not just the question.
    assert "extra_forbidden" in calls[1] or "Extra inputs" in calls[1]


def test_llm_planner_raises_clean_error_after_second_invalid_schema(monkeypatch) -> None:
    def fake_complete(self: LLMPlanner, system: str, user: str) -> str:
        return '{"filters": [], "group_by": [], "metrics": [{"fn": "count", "bogus_field": 1}]}'

    monkeypatch.setattr(LLMPlanner, "complete", fake_complete)
    planner = LLMPlanner(LLMConfig(provider="openai", api_key="test-key", model="test-model"))

    with pytest.raises(LLMResponseError):
        planner.plan("How many rows are there?", {"columns": {}})
