"""Safe dataframe question answering for MCP."""

from mcp_dataframe_qa.engine import DataFrameQA
from mcp_dataframe_qa.schemas import AnalysisPlan, StructuredResult

__all__ = ["AnalysisPlan", "DataFrameQA", "StructuredResult"]

__version__ = "0.1.0"
