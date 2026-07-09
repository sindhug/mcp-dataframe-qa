import argparse
import json
import sys

from mcp_dataframe_qa.config import load_config
from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.scaffold import dataset_slug, write_starter_config


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
        out_path = args.out or f"dataframe_qa_{dataset_slug(args.data)}.yaml"
        written = write_starter_config(args.data, out_path)
        print(f"Wrote {written} ({args.data} detected).")
        print("Add descriptions and synonyms for better answers, then run:")
        print(f"  uv run mcp-dataframe-chat --config {written}")
        return 0

    config = load_config(args.config)
    qa = DataFrameQA.from_config(config, data_path=args.data)

    if args.profile:
        _print_json(qa.profile(config.dataset.id))
        return 0

    if args.ask:
        _print_json(qa.query(args.ask, dataset_id=config.dataset.id).as_dict())
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
