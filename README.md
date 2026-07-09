# MCP DataFrame QA

A research-informed MCP server for safe dataframe question answering over local data.

**MCP DataFrame QA** turns a local CSV, Parquet file, or Pandas dataframe into a natural-language analytics tool that works with MCP-compatible assistants. It ships with a prepared 91,872-row public Zillow Research housing-market dataset, so the repository is useful immediately after cloning while remaining small enough for GitHub and local Pandas.

The project builds on prior work in dataframe question answering, especially [DataFrame QA: A Universal LLM Framework on DataFrame Question Answering Without Data Exposure](https://arxiv.org/abs/2401.15463), and adapts those ideas to a practical, shareable MCP server.

Start the server and ask questions like:

- "What are the top metros by median list price?"
- "Show average median list price by state."
- "How many metro-months had more than 10,000 active listings?"
- "What are the top markets by new listings?"

The design goal is simple: **English in, structured analysis out, no arbitrary code execution by default.**

## Highlights

- Bring your own CSV, Parquet file, or Pandas dataframe
- Ask analytical questions in natural language
- Return structured MCP results in addition to prose
- Expose schema through MCP resources instead of placing raw data in prompts
- Use typed analysis plans instead of raw Python execution
- Enforce read-only execution, timeouts, output caps, and audit logs
- Keep the MCP tool surface small, composable, and easy for models to use

## Dataset Included

The default dataframe is `data/zillow_metro_market.csv`, prepared from public [Zillow Research Housing Data](https://www.zillow.com/research/data/) for-sale listing time series. It combines monthly metro-level and U.S.-level metrics for:

- for-sale inventory
- new listings
- median list price

The prepared table has 91,872 rows, 11 columns, 928 geographies, and month-end observations from 2018-03-31 through 2026-05-31. The CSV is approximately 6.6 MB.

This is not scraped individual listing data. It is aggregated public research data, which makes it a better open-source default: realistic enough to justify dataframe QA, compact enough to commit, and reproducible without browser automation.

To rebuild the dataset from Zillow Research source CSVs:

```bash
python scripts/prepare_zillow_market_data.py
```

See [`data/zillow_metro_market.README.md`](data/zillow_metro_market.README.md) for source URLs, transformation details, and column definitions.

## Why This Exists

The DataFrame QA paper demonstrates that language models can answer dataframe questions by generating Pandas-style analytical queries from dataframe structure, while avoiding direct exposure of the full dataset to the model. That observation is the foundation of this project.

A reusable MCP server, however, has additional engineering requirements: explicit tool schemas, client-visible resources, output validation, execution limits, auditability, and safe defaults for users who clone the repository and bring their own data. This project focuses on that implementation layer.

This repository adopts the following approach:

1. The assistant sees a compact dataframe profile, not your full dataset.
2. Natural language is translated into a typed analysis plan.
3. The plan is validated against the dataframe schema and guardrails.
4. A deterministic read-only executor runs the analysis.
5. Results are returned as structured MCP content plus a concise human-readable answer.

The result is a reusable MCP server for dataframe analytics with explicit production-oriented mechanisms: typed schemas, read-only execution, output caps, timeouts, audit logs, and clear MCP resources/tools/prompts.

## Research Context

This project is motivated by three converging lines of work.

First, dataframe question answering research shows that tabular analysis can be mediated through generated queries rather than full data disclosure. The DataFrame QA framework is particularly relevant because it frames dataframe QA as schema-aware query generation with safe execution.

Second, recent table-QA systems increasingly use multi-stage pipelines: schema understanding, query generation, execution, answer extraction, and refinement. For example, [Agentic LLMs for Question Answering over Tabular Data](https://arxiv.org/abs/2509.09234) reports a natural-language-to-SQL pipeline with verification and iterative refinement for tabular QA.

Third, MCP literature and practice are converging on small, typed, auditable tool surfaces. [MCP Server Architecture Patterns for LLM-Integrated Applications](https://arxiv.org/abs/2606.30317) identifies recurring server patterns such as Resource Gateway, Tool Orchestrator, and Domain-Specific Adapter. MCP DataFrame QA is closest to a Domain-Specific Adapter with Resource Gateway behavior: it exposes dataset profiles as resources and analysis operations as a small number of structured tools.

The contribution of this repository is not a new benchmark result. It is a careful systems design for making dataframe QA easy to clone, inspect, run, and adapt within the MCP ecosystem.

Relative to research prototypes and notebook-oriented dataframe agents, this repository emphasizes:

- a packageable MCP server interface
- explicit resources for dataset profiles
- stable input and output schemas
- a typed intermediate analysis representation
- deterministic read-only execution
- local-first operation with no network access required for analysis
- audit records and result-size controls

## Quick Start

```bash
git clone https://github.com/sindhug/mcp-dataframe-qa.git
cd mcp-dataframe-qa
uv sync
uv run mcp-dataframe-qa
```

Then connect your MCP client to the server.

You can also verify the local engine before connecting an MCP client:

```bash
uv run mcp-dataframe-qa --profile
uv run mcp-dataframe-qa --ask 'What are the top metros by median list price?'
```

Example local MCP configuration:

```json
{
  "mcpServers": {
    "dataframe-qa": {
      "command": "uv",
      "args": [
        "run",
        "mcp-dataframe-qa",
        "--data",
        "/absolute/path/to/your/data.csv"
      ]
    }
  }
}
```

Bring your own dataframe:

```bash
uv run mcp-dataframe-qa --data ~/Downloads/my_data.csv
uv run mcp-dataframe-qa --data ~/Downloads/my_data.parquet
```

Optional project config:

```yaml
# dataframe_qa.yaml
dataset:
  id: default
  path: data/zillow_metro_market.csv
  table_name: zillow_metro_market

limits:
  max_rows_returned: 100
  max_execution_ms: 3000
  max_cell_chars: 500

columns:
  region_name:
    description: Metro or national region name
    semantic_type: dimension
    synonyms: [metro, market, region, msa]
  state_name:
    description: State abbreviation for the metro area
    semantic_type: dimension
    synonyms: [state, state code]
  for_sale_inventory:
    description: Count of unique listings active at any time during the month
    semantic_type: count
    synonyms: [inventory, active listings, homes for sale]
  median_list_price:
    description: Median listed price in USD
    semantic_type: currency
    synonyms: [median price, list price, asking price, price]
```

## What You Get

### MCP Resources

Resources expose dataset context without dumping raw data into the model.

```text
dataframe://default/profile
dataframe://default/columns
dataframe://default/examples
```

### MCP Tools

The public tool surface stays intentionally small.

```python
query_dataframe(question: str, dataset_id: str = "default") -> StructuredResult
execute_analysis_plan(plan: AnalysisPlan, dataset_id: str = "default") -> StructuredResult
preview_dataframe(dataset_id: str = "default", limit: int = 20) -> StructuredResult
```

`query_dataframe` is the ergonomic entry point. `execute_analysis_plan` is the stable core. `preview_dataframe` is capped and meant for orientation, not data export.

### MCP Prompts

Prompts make common workflows discoverable in clients that support them.

```text
ask_dataframe
explain_dataframe
```

## The Core Contract

The DataFrame QA paper studies LLM-generated Pandas queries as a general dataframe QA mechanism. MCP DataFrame QA preserves the central idea of translating natural language into executable analysis, but uses a typed intermediate representation by default.

Instead of executing arbitrary Python, the assistant produces a typed `AnalysisPlan`.

```json
{
  "filters": [
    {
      "column": "price",
      "op": "<",
      "value": 1000000
    }
  ],
  "group_by": ["bedrooms"],
  "metrics": [
    {
      "fn": "avg",
      "column": "price",
      "as": "avg_price"
    }
  ],
  "sort": [
    {
      "column": "bedrooms",
      "direction": "asc"
    }
  ],
  "limit": 100
}
```

The server validates that plan before anything runs:

- columns must exist
- operations must be allowed
- result limits are enforced
- execution is read-only
- generated SQL is constrained to safe statements
- large outputs are summarized or rejected
- every tool call receives an audit id

This plan-based layer is the main engineering adaptation. It makes the generated analysis easier to validate, explain, test, cache, and audit before execution.

Example structured result:

```json
{
  "answer": "There are 348 listings under $1M.",
  "kind": "scalar",
  "value": 348,
  "table": null,
  "chart": null,
  "warnings": [],
  "audit_id": "qry_20260709_0001"
}
```

## Design Principles

### 1. Dataframe Profile as Context

The model gets schema, column descriptions, safe samples, and summary statistics. It does not need the full dataframe to answer most analytical questions.

This follows the direction of DataFrame QA research: generate analysis from dataframe structure while minimizing dataset exposure.

### 2. Typed Plans as Execution Contracts

Generated Pandas remains a valuable research and prototyping technique. For a shareable MCP server, this project uses a typed plan as the default because it provides a narrower execution contract and clearer validation boundary.

### 3. Minimal Tool Surface

MCP servers are easier for models and humans to understand when the tool list is small and composable. This project exposes a few high-leverage tools instead of a tool per dataframe operation.

### 4. Structured Results First

Every answer returns machine-readable content. The prose is there for humans, but downstream agents and apps should be able to consume the result directly.

### 5. Safe by Default, Extensible by Choice

The default path is read-only analysis. Advanced code execution, custom functions, remote data loading, or broader filesystem access should be explicit opt-ins.

## Scope and Limitations

MCP DataFrame QA is designed for practical, local dataframe analysis. It intentionally does not attempt to solve every form of tabular reasoning.

- It is best suited to single-table or lightly configured dataframe workflows.
- It prioritizes deterministic aggregates, filters, sorting, grouping, and summaries.
- It does not replace a governed enterprise semantic layer.
- It does not guarantee correct answers for ambiguous business terminology without column descriptions or synonyms.
- It does not expose unrestricted Python execution in the default path.
- It treats charting as a structured output problem, not as a primary visualization framework.

These constraints are deliberate. They keep the repository small enough to clone and understand, while leaving room for opt-in extensions.

## Guardrails

Guardrails are treated as part of the core interface rather than as optional deployment details.

- Read-only dataframe access
- No filesystem writes during analysis
- No network access from analysis execution
- No arbitrary imports in the default path
- Execution timeouts
- Row, cell, and payload-size caps
- Query validation before execution
- Output sanitization
- Audit logs for every tool call
- Stable JSON schemas for tool input and output
- Optional column allowlists and deny lists

The default server should be safe enough to run locally on private datasets and simple enough for analysts to understand.

## Architecture

```text
User question
    |
    v
MCP tool: query_dataframe
    |
    v
Dataset registry + dataframe profile
    |
    v
Planner: question -> AnalysisPlan
    |
    v
Validator: schema, limits, policy
    |
    v
Executor: read-only SQL/DataFrame engine
    |
    v
StructuredResult: answer, value/table/chart, audit id
```

Recommended implementation layout:

```text
src/mcp_dataframe_qa/
  server.py          # MCP resources, tools, prompts
  datasets.py        # dataset registry and loaders
  profiling.py       # schema, stats, safe examples
  planner.py         # question -> AnalysisPlan
  schemas.py         # Pydantic models
  validator.py       # guardrails and policy checks
  executor.py        # read-only DataFrame execution
  results.py         # structured result formatting
  audit.py           # audit records
```

## Example Questions

Bundled Zillow Research dataset:

- "What are the top metros by median list price?"
- "Show average median list price by state."
- "How many metro-months had more than 10,000 active listings?"
- "Show average for-sale inventory by state."
- "What are the top markets by new listings?"
- "How many rows have median list price over $1M?"

Bring your own listing dataframe:

- "How many homes are under $750k?"
- "Show average price by bedroom count."
- "How many listings have at least 3 bedrooms and 2 bathrooms?"
- "What are the top ZIP codes by median price?"

Sales:

- "What was revenue by month?"
- "Which region had the highest average order value?"
- "Show the top 20 customers by total spend."
- "How many orders were refunded last quarter?"

Product analytics:

- "What is the conversion rate by plan?"
- "Which acquisition channel has the highest retention?"
- "Show weekly active users by cohort."
- "How many accounts have not logged in for 30 days?"

## When to Use This

Use MCP DataFrame QA when:

- you have one dataframe or a small set of local tabular files
- you want natural-language analytics inside an MCP-compatible assistant
- you care about privacy, auditability, and predictable execution
- you want a reusable adapter instead of a one-off notebook

Do not use it as a replacement for:

- a full semantic BI layer
- governed enterprise data warehouses
- multi-tenant production analytics without additional authorization
- unrestricted Python notebooks

## Current Status

- CSV and JSON loading
- Parquet loading through Pandas when a Parquet engine is installed
- Bundled public Zillow Research housing-market dataframe
- Reproducible Zillow dataset preparation script
- Configurable column descriptions, semantic types, and synonyms
- Dataset profiling with capped examples
- Pydantic `AnalysisPlan` schemas
- Conservative built-in natural-language planner for common questions
- Read-only Pandas-backed execution
- MCP resources for dataset profiles
- MCP tools with structured result payloads
- Audit IDs and optional JSONL audit log sink
- Test coverage for common real-estate questions and guardrails

## Roadmap

- DuckDB-backed execution adapter
- Direct Pandas dataframe registration examples
- MCP tool annotations for clients that surface read-only hints
- CLI dataset profiler
- Optional chart specs
- Optional local-only Pandas code sandbox for advanced users
- Evaluation set for common dataframe questions
- Larger table-QA regression suite

## Contributing

This project is intentionally small and opinionated. Good contributions usually make it easier to trust, install, test, or adapt:

- new dataframe loaders
- stronger plan validation
- better profiling and semantic column hints
- more evaluation questions
- clearer MCP client examples
- safer execution limits
- sharper documentation

The guiding rule: keep the default path simple, read-only, predictable, and reliable.

## Standards and References

This project is shaped by current MCP and table-QA patterns:

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification/2025-11-25)
- [MCP tools: structured content and output schemas](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- [MCP resources](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)
- [MCP security best practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)
- [Zillow Research Housing Data](https://www.zillow.com/research/data/)
- [DataFrame QA: A Universal LLM Framework on DataFrame Question Answering Without Data Exposure](https://arxiv.org/abs/2401.15463)
- [MCP Server Architecture Patterns for LLM-Integrated Applications](https://arxiv.org/abs/2606.30317)
- [Model Context Protocol Threat Modeling and Tool Poisoning Analysis](https://arxiv.org/abs/2603.22489)

## Philosophy

The intended user experience is deliberately simple:

1. Drop in a file.
2. Start the MCP server.
3. Ask a question.
4. Get a correct, capped, auditable answer.

The project avoids large prompt payloads, opaque Python execution, and unnecessary dataset exposure. It aims to provide a small, inspectable adapter between a dataframe and the assistant a user already uses.

## License

MIT
