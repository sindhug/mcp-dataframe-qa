import argparse
import json
import sys

from mcp_dataframe_qa.config import load_config
from mcp_dataframe_qa.datasets import load_dataframe
from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.llm import (
    LLMConfigurationError,
    LLMPlanner,
    LLMResponseError,
    load_env_file,
    resolve_llm_config,
)
from mcp_dataframe_qa.scaffold import dataset_slug, describe_columns_with_llm, write_starter_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-dataframe-qa",
        description="Run a safe MCP dataframe question-answering server.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", help="Path to dataframe_qa.yaml.")
    parser.add_argument("--data", help="Path to a CSV, Parquet, or JSON dataframe.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print the dataset profile and exit.",
    )
    parser.add_argument("--ask", help="Ask one local question without starting MCP.")
    parser.add_argument(
        "--planner",
        choices=["llm", "heuristic"],
        default="llm",
        help=(
            "Planner for --ask: llm (default, uses the configured LLM provider) or "
            "heuristic (built-in deterministic planner, no API key needed)."
        ),
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help=(
            "Write a starter YAML config for --data, with every column detected "
            "and ready to annotate, then exit."
        ),
    )
    parser.add_argument(
        "--out",
        help="Output path for --init-config. Defaults to dataframe_qa_<name>.yaml.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "For --init-config, skip LLM-assisted column descriptions and use the "
            "basic name-based heuristic instead, even if a provider is configured."
        ),
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        help="MCP transport to use when starting the server.",
    )
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.init_config:
        if not args.data:
            print("Error: --init-config requires --data <path>.", file=sys.stderr)
            return 2

        column_info = None
        if not args.no_llm:
            load_env_file()
            try:
                llm_config = resolve_llm_config()
            except LLMConfigurationError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                print(
                    "Pass --no-llm to scaffold with the basic heuristic instead.",
                    file=sys.stderr,
                )
                return 2
            print(f"Drafting column descriptions with {llm_config.provider}...")
            try:
                frame = load_dataframe(args.data)
                column_info = describe_columns_with_llm(frame, llm_config)
            except LLMResponseError as exc:
                print(f"Warning: LLM description drafting failed ({exc}).", file=sys.stderr)
                print("Falling back to the basic heuristic.", file=sys.stderr)
                column_info = None

        out_path = args.out or f"dataframe_qa_{dataset_slug(args.data)}.yaml"
        written = write_starter_config(args.data, out_path, column_info=column_info)
        print(f"Wrote {written} ({args.data} detected).")
        if column_info:
            print("Review the drafted descriptions and synonyms, then run:")
        else:
            print("Add descriptions and synonyms for better answers, then run:")
        print(f"  uv run mcp-dataframe-chat --config {written}")
        return 0

    config = load_config(args.config)
    qa = DataFrameQA.from_config(config, data_path=args.data)

    if args.profile:
        _print_json(qa.profile(config.dataset.id))
        return 0

    if args.ask:
        if args.planner == "heuristic":
            result = qa.query(args.ask, dataset_id=config.dataset.id)
        else:
            load_env_file()
            try:
                llm_config = resolve_llm_config()
            except LLMConfigurationError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                print(
                    "Pass --planner heuristic to use the built-in deterministic planner instead.",
                    file=sys.stderr,
                )
                return 2
            profile = qa.profile(config.dataset.id)
            try:
                plan = LLMPlanner(llm_config).plan(args.ask, profile)
            except LLMResponseError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 3
            result = qa.execute_plan(plan, dataset_id=config.dataset.id)
            if result.answer and "Query:" not in result.answer:
                result.answer = f"{result.answer} Query: {args.ask}"
        _print_json(result.as_dict())
        return 0

    try:
        from mcp_dataframe_qa.server import build_server
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    server = build_server(qa)
    try:
        server.run(transport=args.transport)
    except TypeError:
        server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
