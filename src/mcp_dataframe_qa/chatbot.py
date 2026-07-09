import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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
    return parser


def extract_payload(tool_result: Any) -> Dict[str, Any]:
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
        return "%.2f" % value
    return str(value)


def print_table(columns: List[str], rows: List[Dict[str, Any]], max_rows: int) -> None:
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
        print("... %d more row(s) capped by local display" % remaining)


def print_answer(payload: Dict[str, Any], max_rows: int, show_plan: bool) -> None:
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
        print("Warning: %s" % warning)

    if show_plan and payload.get("plan"):
        print()
        print(json.dumps(payload["plan"], indent=2, sort_keys=True))
    print()


async def ask(session: ClientSession, question: str, rows: int, show_plan: bool) -> None:
    result = await session.call_tool("query_dataframe", {"question": question})
    payload = extract_payload(result)
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
    root = Path.cwd()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

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
                    await ask(session, args.question, rows=args.rows, show_plan=args.show_plan)
                    return

                print("MCP DataFrame QA local chatbot")
                print("Ask a question about the dataframe. Type 'exit' to stop.")
                print()
                for question in prompt_lines():
                    await ask(session, question, rows=args.rows, show_plan=args.show_plan)


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    anyio.run(run_chatbot, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
