"""End-to-end tests of the real LLM planning path.

These hit the actual configured LLM provider by default: costs real money
and requires an API key in .env, the same as any other real use of this
tool. Run `pytest tests/test_llm_live.py --planner=heuristic` to swap to
the free, built-in deterministic planner instead (no API key required).

Assertions here check the *outcome* of each question, not the exact shape
of the generated plan. Unlike the heuristic planner, the LLM isn't
guaranteed to produce a byte-identical plan across runs even when it
answers correctly, so these deliberately assert on the answer, not on
plan internals.
"""

import pytest

from mcp_dataframe_qa.config import load_config
from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.llm import (
    LLMConfigurationError,
    LLMPlanner,
    load_env_file,
    resolve_llm_config,
)
from mcp_dataframe_qa.schemas import StructuredResult

ZILLOW_CONFIG = "dataframe_qa.yaml"


def _qa() -> DataFrameQA:
    return DataFrameQA.from_config(load_config(ZILLOW_CONFIG))


def _ask(qa: DataFrameQA, question: str, planner_mode: str) -> StructuredResult:
    if planner_mode == "heuristic":
        return qa.query(question)

    load_env_file()
    try:
        llm_config = resolve_llm_config()
    except LLMConfigurationError:
        pytest.skip(
            "No LLM provider configured; set an API key in .env or pass --planner=heuristic."
        )
    profile = qa.profile("default")
    plan = LLMPlanner(llm_config).plan(question, profile)
    return qa.execute_plan(plan)


def test_top_metros_by_median_list_price(planner_mode) -> None:
    result = _ask(_qa(), "What are the top metros by median list price?", planner_mode)
    assert result.kind == "table"
    assert result.table is not None
    assert len(result.table.rows) > 0
    assert "median_list_price" in result.table.columns


def test_count_of_metro_months_with_large_inventory(planner_mode) -> None:
    question = "How many metro-months had more than 10,000 active listings?"
    result = _ask(_qa(), question, planner_mode)
    assert result.kind == "scalar"
    assert isinstance(result.value, int)
    assert result.value > 0


def test_average_inventory_by_state(planner_mode) -> None:
    result = _ask(_qa(), "Show average for-sale inventory by state.", planner_mode)
    assert result.kind == "table"
    assert result.table is not None
    assert len(result.table.rows) > 0


def test_nonexistent_column_question_is_reported_as_an_error(planner_mode) -> None:
    """The LLM should recognize an unanswerable question rather than guess.

    The heuristic planner has no "I don't know" path: its column-selection
    fallback always returns some numeric column, so it would silently answer
    this with a guess instead of an error. That's LLM-only behavior, so this
    test only applies in llm mode.
    """
    if planner_mode == "heuristic":
        pytest.skip(
            "The heuristic planner always guesses a column instead of recognizing "
            "an unanswerable question; this behavior is LLM-only."
        )
    result = _ask(_qa(), "What is the average of a column that does not exist", planner_mode)
    # The LLM isn't perfectly deterministic here: one run may return an empty
    # metrics list ("must include at least one metric"), another may invent a
    # literal column name from the question text ("Unknown metric column
    # 'column that does not exist'"). Both are legitimate, honest validation
    # errors -- what actually matters is that it errors instead of silently
    # returning a confident guess, not which specific rule catches it.
    assert result.kind == "error"
