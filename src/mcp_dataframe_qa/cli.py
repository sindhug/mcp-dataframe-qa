import argparse
import json
import sys
from typing import Optional

from mcp_dataframe_qa.config import load_config
from mcp_dataframe_qa.engine import DataFrameQA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-dataframe-qa",
        description="Run a safe MCP dataframe question-answering server.",
    )
    parser.add_argument("--config", help="Path to dataframe_qa.yaml.")
    parser.add_argument("--data", help="Path to a CSV, Parquet, or JSON dataframe.")
    parser.add_argument("--profile", action="store_true", help="Print the dataset profile and exit.")
    parser.add_argument("--ask", help="Ask one local question without starting MCP.")
    parser.add_argument("--transport", default="stdio", help="MCP transport to use when starting the server.")
    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
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
