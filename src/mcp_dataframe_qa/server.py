import json
from typing import Any, Dict

from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.schemas import AnalysisPlan

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise RuntimeError(
        "The MCP Python SDK is required to run the server. Install with `uv sync` "
        "or `pip install mcp-dataframe-qa`."
    ) from exc


def build_server(qa: DataFrameQA) -> FastMCP:
    mcp = FastMCP("MCP DataFrame QA")

    @mcp.resource("dataframe://{dataset_id}/profile")
    def dataframe_profile(dataset_id: str = "default") -> str:
        """Return a compact profile of the dataframe schema and safe examples."""
        return json.dumps(qa.profile(dataset_id), indent=2, sort_keys=True, default=str)

    @mcp.resource("dataframe://{dataset_id}/columns")
    def dataframe_columns(dataset_id: str = "default") -> str:
        """Return column metadata for a dataframe."""
        profile = qa.profile(dataset_id)
        return json.dumps(profile["columns"], indent=2, sort_keys=True, default=str)

    @mcp.resource("dataframe://{dataset_id}/examples")
    def dataframe_examples(dataset_id: str = "default") -> str:
        """Return a capped set of example rows."""
        profile = qa.profile(dataset_id)
        return json.dumps(profile["examples"], indent=2, sort_keys=True, default=str)

    @mcp.tool()
    def query_dataframe(question: str, dataset_id: str = "default") -> Dict[str, Any]:
        """Answer a natural-language question about the dataframe.

        This tool uses the built-in conservative planner. For complex analysis,
        call execute_analysis_plan with a typed plan.
        """
        return qa.query(question, dataset_id=dataset_id).as_dict()

    @mcp.tool()
    def execute_analysis_plan(plan: Dict[str, Any], dataset_id: str = "default") -> Dict[str, Any]:
        """Execute a validated, read-only dataframe analysis plan."""
        parsed = AnalysisPlan.model_validate(plan)
        return qa.execute_plan(parsed, dataset_id=dataset_id).as_dict()

    @mcp.tool()
    def preview_dataframe(dataset_id: str = "default", limit: int = 20) -> Dict[str, Any]:
        """Return a capped preview of the dataframe."""
        return qa.preview(dataset_id=dataset_id, limit=limit).as_dict()

    @mcp.prompt()
    def ask_dataframe(question: str, dataset_id: str = "default") -> str:
        return (
            "Use dataframe://%s/profile for schema context. Then answer this question "
            "with query_dataframe or execute_analysis_plan: %s" % (dataset_id, question)
        )

    @mcp.prompt()
    def explain_dataframe(dataset_id: str = "default") -> str:
        return (
            "Read dataframe://%s/profile and summarize what analytical questions this "
            "dataframe appears suited to answer. Do not infer facts beyond the profile."
            % dataset_id
        )

    return mcp
