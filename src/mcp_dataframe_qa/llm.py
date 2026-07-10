import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_dataframe_qa.schemas import AnalysisPlan

PROVIDERS = {"openai", "anthropic", "gemini"}
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
MODEL_ENV = {
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "gemini": "GEMINI_MODEL",
}
DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-2.5-flash",
}


class LLMConfigurationError(RuntimeError):
    """Raised when the chatbot cannot determine a usable LLM provider configuration."""


class LLMResponseError(RuntimeError):
    """Raised when an LLM provider response cannot be used as an AnalysisPlan."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str
    timeout_seconds: int = 60


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_llm_config(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
) -> LLMConfig:
    env = env or os.environ
    selected_provider = (provider or env.get("LLM_PROVIDER") or "").strip().lower()

    if not selected_provider:
        for candidate in ("openai", "anthropic", "gemini"):
            if env.get(API_KEY_ENV[candidate]):
                selected_provider = candidate
                break

    if selected_provider not in PROVIDERS:
        raise LLMConfigurationError(
            "Set LLM_PROVIDER to one of: openai, anthropic, gemini. "
            "Then set the matching API key in .env."
        )

    selected_key = api_key or env.get(API_KEY_ENV[selected_provider])
    if not selected_key:
        key_name = API_KEY_ENV[selected_provider]
        raise LLMConfigurationError(
            f"No API key found for provider '{selected_provider}'. "
            f"Copy .env.example to .env and set {key_name}."
        )

    selected_model = (
        model
        or env.get("LLM_MODEL")
        or env.get(MODEL_ENV[selected_provider])
        or DEFAULT_MODELS[selected_provider]
    )
    timeout = int(env.get("LLM_TIMEOUT_SECONDS", "60"))
    return LLMConfig(
        provider=selected_provider,
        api_key=selected_key,
        model=selected_model,
        timeout_seconds=timeout,
    )


def compact_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    columns = {}
    for name, metadata in profile.get("columns", {}).items():
        columns[name] = {
            key: metadata.get(key)
            for key in ["dtype", "description", "semantic_type", "synonyms", "stats", "top_values"]
            if metadata.get(key) is not None
        }
    return {
        "dataset_id": profile.get("dataset_id"),
        "table_name": profile.get("table_name"),
        "row_count": profile.get("row_count"),
        "column_count": profile.get("column_count"),
        "columns": columns,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()

    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Model did not return valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError("Model returned JSON, but not a JSON object.")
    return parsed


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LLMResponseError(f"Provider HTTP {exc.code}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise LLMResponseError(f"Provider request failed: {exc.reason}") from exc


def _prompt(question: str, profile: Mapping[str, Any]) -> dict[str, str]:
    system = (
        "You translate dataframe questions into a JSON AnalysisPlan. "
        "Return only a JSON object. Do not include markdown or prose. "
        "Use only columns present in the dataframe profile. "
        "Allowed filter ops: ==, !=, <, <=, >, >=, in, not_in, contains. "
        "Allowed metric functions: count, sum, avg, mean, median, min, max, nunique. "
        "For custom numeric measures such as ratios, use the optional derive list. "
        "Allowed expression ops are column, literal, add, subtract, multiply, divide, ratio, "
        "==, !=, <, <=, >, >=. "
        "For rate or proportion questions (for example 'how often does X happen'), derive a "
        "0/1 indicator column with a comparison op, then take its avg. "
        "Derived expressions must be JSON trees, never Python code. "
        "For row counts use {\"fn\":\"count\",\"column\":\"*\",\"as\":\"count\"}. "
        "For top-N questions, set group_by, one metric, descending sort on the metric alias, "
        "and limit to the requested N or 10 if no N is specified."
    )
    user = json.dumps(
        {
            "question": question,
            "dataframe_profile": compact_profile(profile),
            "analysis_plan_shape": {
                "derive": [
                    {
                        "name": "derived_metric",
                        "expr": {
                            "op": "divide",
                            "left": {"op": "column", "column": "numeric_column"},
                            "right": {"op": "column", "column": "other_numeric_column"},
                        },
                    }
                ],
                "filters": [{"column": "column_name", "op": ">", "value": 10}],
                "group_by": ["column_name"],
                "metrics": [{"fn": "median", "column": "column_name", "as": "metric_alias"}],
                "sort": [{"column": "metric_alias_or_group_column", "direction": "desc"}],
                "limit": 10,
            },
        },
        sort_keys=True,
    )
    return {"system": system, "user": user}


class LLMPlanner:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def plan(self, question: str, profile: Mapping[str, Any]) -> AnalysisPlan:
        prompts = _prompt(question, profile)
        text = self.complete(prompts["system"], prompts["user"])
        payload = extract_json_object(text)
        return AnalysisPlan.model_validate(payload)

    def complete(self, system: str, user: str) -> str:
        if self.config.provider == "openai":
            return self._complete_openai(system, user)
        if self.config.provider == "anthropic":
            return self._complete_anthropic(system, user)
        if self.config.provider == "gemini":
            return self._complete_gemini(system, user)
        raise LLMConfigurationError(f"Unsupported provider: {self.config.provider}")

    def _complete_openai(self, system: str, user: str) -> str:
        response = _post_json(
            "https://api.openai.com/v1/responses",
            {"Authorization": f"Bearer {self.config.api_key}"},
            {
                "model": self.config.model,
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.config.timeout_seconds,
        )
        if response.get("output_text"):
            return str(response["output_text"])
        for item in response.get("output", []):
            for content in item.get("content", []):
                if "text" in content:
                    return str(content["text"])
        raise LLMResponseError("OpenAI response did not contain output text.")

    def _complete_anthropic(self, system: str, user: str) -> str:
        response = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
            },
            {
                "model": self.config.model,
                "max_tokens": 1200,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=self.config.timeout_seconds,
        )
        for block in response.get("content", []):
            if block.get("type") == "text" and block.get("text"):
                return str(block["text"])
        raise LLMResponseError("Anthropic response did not contain text content.")

    def _complete_gemini(self, system: str, user: str) -> str:
        model = self.config.model
        if model.startswith("models/"):
            model = model.removeprefix("models/")
        quoted_model = urllib.parse.quote(model)
        query = urllib.parse.urlencode({"key": self.config.api_key})
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quoted_model}:generateContent?{query}"
        )
        response = _post_json(
            url,
            {},
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                },
            },
            timeout=self.config.timeout_seconds,
        )
        for candidate in response.get("candidates", []):
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            texts = [part.get("text", "") for part in parts if part.get("text")]
            if texts:
                return "\n".join(texts)
        raise LLMResponseError("Gemini response did not contain text content.")
