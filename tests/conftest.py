import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--planner",
        action="store",
        default="llm",
        choices=("llm", "heuristic"),
        help=(
            "Planner used by tests that ask real natural-language questions "
            "(tests/test_llm_live.py): llm (default, hits the configured LLM "
            "provider, requires an API key in .env and costs real money) or "
            "heuristic (built-in deterministic planner, free, no API key needed)."
        ),
    )


@pytest.fixture
def planner_mode(request) -> str:
    return request.config.getoption("--planner")
