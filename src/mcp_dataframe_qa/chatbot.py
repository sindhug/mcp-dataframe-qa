import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_dataframe_qa.llm import (
    LLMConfigurationError,
    LLMPlanner,
    LLMResponseError,
    load_env_file,
    resolve_llm_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-dataframe-chat",
        description=(
            "Run a local terminal chatbot that talks to the MCP DataFrame QA server over stdio."
        ),
    )
    parser.add_argument(
        "--config",
        default="dataframe_qa.yaml",
        help="Path to the MCP DataFrame QA config file.",
    )
    parser.add_argument(
        "--question",
        help="Ask one question and exit. If omitted, starts an interactive loop.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Maximum number of table rows to print for each answer.",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the generated analysis plan after each answer.",
    )
    parser.add_argument(
        "--planner",
        choices=["llm", "heuristic"],
        default=os.environ.get("MCP_DFQA_PLANNER", "llm"),
        help="Use the model-backed planner or the built-in heuristic planner.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        help=(
            "LLM provider for the model-backed planner. Defaults to LLM_PROVIDER "
            "or key auto-detection."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Model name for the selected provider. Defaults to LLM_MODEL or "
            "provider-specific env vars."
        ),
    )
    parser.add_argument(
        "--api-key",
        help=(
            "API key for the selected provider. Prefer .env or environment variables "
            "for normal use."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        default="default",
        help="Dataset id to query through the MCP server.",
    )
    return parser


def extract_payload(tool_result: Any) -> dict[str, Any]:
    structured = tool_result.structuredContent or {}
    if isinstance(structured, dict) and isinstance(structured.get("result"), dict):
        return structured["result"]

    for block in tool_result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return {"kind": "error", "answer": "Tool returned no structured dataframe result."}


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def print_table(columns: list[str], rows: list[dict[str, Any]], max_rows: int) -> None:
    visible_rows = rows[:max_rows]
    widths = {
        column: max(
            len(column),
            *(len(format_value(row.get(column))) for row in visible_rows),
        )
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for row in visible_rows:
        print(" | ".join(format_value(row.get(column)).ljust(widths[column]) for column in columns))

    remaining = len(rows) - len(visible_rows)
    if remaining > 0:
        print(f"... {remaining} more row(s) capped by local display")


def print_answer(payload: dict[str, Any], max_rows: int, show_plan: bool) -> None:
    print()
    print(payload.get("answer", "No answer returned."))

    table = payload.get("table")
    if isinstance(table, dict) and table.get("rows"):
        print()
        print_table(table.get("columns", []), table.get("rows", []), max_rows=max_rows)

    if payload.get("value") is not None:
        print()
        print(format_value(payload["value"]))

    warnings = payload.get("warnings") or []
    for warning in warnings:
        print(f"Warning: {warning}")

    if show_plan and payload.get("plan"):
        print()
        print(json.dumps(payload["plan"], indent=2, sort_keys=True))
    print()


async def read_profile(session: ClientSession, dataset_id: str) -> dict[str, Any]:
    resource = await session.read_resource(f"dataframe://{dataset_id}/profile")
    for content in resource.contents:
        text = getattr(content, "text", None)
        if text:
            return json.loads(text)
    raise RuntimeError("MCP server returned no dataframe profile resource content.")


async def ask_with_heuristic(
    session: ClientSession,
    question: str,
    dataset_id: str,
    rows: int,
    show_plan: bool,
) -> None:
    result = await session.call_tool(
        "query_dataframe",
        {"question": question, "dataset_id": dataset_id},
    )
    payload = extract_payload(result)
    print_answer(payload, max_rows=rows, show_plan=show_plan)


async def ask_with_llm(
    session: ClientSession,
    planner: LLMPlanner,
    question: str,
    dataset_id: str,
    rows: int,
    show_plan: bool,
) -> None:
    profile = await read_profile(session, dataset_id)
    plan = planner.plan(question, profile)
    result = await session.call_tool(
        "execute_analysis_plan",
        {"plan": plan.model_dump(by_alias=True, exclude_none=True), "dataset_id": dataset_id},
    )
    payload = extract_payload(result)
    if payload.get("answer") and "Query:" not in payload["answer"]:
        payload["answer"] = f"{payload['answer']} Query: {question}"
    if show_plan and not payload.get("plan"):
        payload["plan"] = plan.model_dump(by_alias=True, exclude_none=True)
    print_answer(payload, max_rows=rows, show_plan=show_plan)


def prompt_lines() -> Iterable[str]:
    while True:
        try:
            question = input("dataframe> ").strip()
        except EOFError:
            return
        if question.lower() in {"exit", "quit", ":q"}:
            return
        if question:
            yield question


async def run_chatbot(args: argparse.Namespace) -> None:
    load_env_file()
    root = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    planner = None
    if args.planner == "llm":
        planner = LLMPlanner(
            resolve_llm_config(provider=args.provider, model=args.model, api_key=args.api_key)
        )

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_dataframe_qa.cli", "--config", str(config_path)],
        cwd=str(root),
    )

    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(server, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                if args.question:
                    if planner:
                        await ask_with_llm(
                            session,
                            planner,
                            args.question,
                            dataset_id=args.dataset_id,
                            rows=args.rows,
                            show_plan=args.show_plan,
                        )
                    else:
                        await ask_with_heuristic(
                            session,
                            args.question,
                            dataset_id=args.dataset_id,
                            rows=args.rows,
                            show_plan=args.show_plan,
                        )
                    return

                print("MCP DataFrame QA local chatbot")
                if planner:
                    print(f"Using {planner.config.provider} model {planner.config.model}.")
                else:
                    print("Using built-in heuristic planner.")
                print("Ask a question about the dataframe. Type 'exit' to stop.")
                print()
                for question in prompt_lines():
                    if planner:
                        await ask_with_llm(
                            session,
                            planner,
                            question,
                            dataset_id=args.dataset_id,
                            rows=args.rows,
                            show_plan=args.show_plan,
                        )
                    else:
                        await ask_with_heuristic(
                            session,
                            question,
                            dataset_id=args.dataset_id,
                            rows=args.rows,
                            show_plan=args.show_plan,
                        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        anyio.run(run_chatbot, args)
        return 0
    except LLMConfigurationError as exc:
        print(f"LLM configuration error: {exc}", file=sys.stderr)
        return 2
    except LLMResponseError as exc:
        print(f"LLM response error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
