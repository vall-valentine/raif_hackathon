"""OpenRouter baseline for session-level red-flag classification."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import typing

import httpx

LOGGER = logging.getLogger("uvicorn.error")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"
DEFAULT_PROMPT_PATH = pathlib.Path(__file__).with_name("prompts") / "red_flag_classifier.md"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 2000
DEFAULT_THINKING_BUDGET = 1500

CLEAN_LABEL = "clean"
RED_FLAG_CATEGORIES: set[str] = {
    "information_extraction",
    "transaction_coercion",
    "policy_manipulation",
    "identity_deception",
    "adversarial_attack",
    "scope_violation",
}

JsonObject = dict[str, typing.Any]
RED_FLAG_RESPONSE_FORMAT: JsonObject = {
    "type": "json_schema",
    "json_schema": {
        "name": "red_flag_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Session-level red-flag label.",
                    "enum": [
                        "clean",
                        "information_extraction",
                        "transaction_coercion",
                        "policy_manipulation",
                        "identity_deception",
                        "adversarial_attack",
                        "scope_violation",
                    ],
                },
            },
            "required": ["category"],
            "additionalProperties": False,
        },
    },
}


@typing.final
class LLMClient:
    """Small OpenRouter chat-completions client."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.timeout_seconds = _read_float_env("OPENROUTER_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        self.prompt_text = _load_prompt()

    def process_dialogue_classification(self, dialogue_text: str) -> str | None:
        return parse_red_flag_category(self.request_completion(dialogue_text))

    def request_completion(self, dialogue_text: str) -> str | None:
        if not self.api_key:
            return None

        request_payload: JsonObject = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt_text},
                {"role": "user", "content": f"Диалог для классификации:\n\n{dialogue_text}"},
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "thinking": {
                "type": "enabled",
                "budget_tokens": DEFAULT_THINKING_BUDGET,
            },
            "response_format": RED_FLAG_RESPONSE_FORMAT,
        }

        try:
            response = httpx.post(
                OPENROUTER_API_URL,
                headers=_build_openrouter_headers(self.api_key),
                json=request_payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as request_error:
            LOGGER.warning("OpenRouter request failed: %s", request_error)
            return None

        return _extract_response_content(response)


def parse_red_flag_category(completion_text: str | None) -> str | None:
    response_data = _parse_json_object(completion_text)
    if response_data is None:
        return None

    category = _extract_category_value(response_data)
    if category is None:
        return None

    normalized_category = category.strip().lower()
    if normalized_category in {CLEAN_LABEL, "", "none", "null"}:
        return None
    if normalized_category in RED_FLAG_CATEGORIES:
        return normalized_category

    LOGGER.warning("OpenRouter returned unsupported red-flag category: %s", category)
    return None


def process_risk_detection(
    llm_client: LLMClient,
    messages: str,
) -> dict[str, str] | None:
    """Classifies one already formatted dialogue and returns evaluator-compatible output."""
    category = llm_client.process_dialogue_classification(messages)
    if category is None:
        return None
    return {"category": category}


def load_llm() -> LLMClient:
    """Create an OpenRouter client at application startup."""
    return LLMClient()


def _load_prompt() -> str:
    prompt_path = pathlib.Path(os.getenv("RED_FLAG_PROMPT_PATH", DEFAULT_PROMPT_PATH))
    try:
        return prompt_path.read_text(encoding="utf-8")
    except OSError as prompt_error:
        LOGGER.warning("Could not read prompt file %s: %s", prompt_path, prompt_error)
        return "Classify the dialogue as clean or one red-flag category. Return JSON with key category."


def _build_openrouter_headers(openrouter_api_key: str) -> dict[str, str]:
    request_headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }
    referer_header = os.getenv("OPENROUTER_SITE_URL", "").strip()
    app_name = os.getenv("OPENROUTER_APP_NAME", "Red Flag Detector").strip()
    if referer_header:
        request_headers["HTTP-Referer"] = referer_header
    if app_name:
        request_headers["X-Title"] = app_name
    return request_headers


def _read_float_env(variable_name: str, default_value: float) -> float:
    raw_value = os.getenv(variable_name, "").strip()
    if not raw_value:
        return default_value
    try:
        return float(raw_value)
    except ValueError:
        LOGGER.warning("Invalid %s=%r, using %s", variable_name, raw_value, default_value)
        return default_value


def _extract_message_content(response_data: JsonObject) -> str | None:
    completion_choices = response_data.get("choices")
    if not isinstance(completion_choices, list) or not completion_choices:
        return None

    first_choice = completion_choices[0]
    if not isinstance(first_choice, dict):
        return None

    choice_message = first_choice.get("message")
    if not isinstance(choice_message, dict):
        return None

    message_content = choice_message.get("content")
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        return "\n".join(
            one_text_block
            for one_text_block in (_extract_text_block(one_content_block) for one_content_block in message_content)
            if one_text_block
        )
    return None


def _extract_response_content(response: httpx.Response) -> str | None:
    try:
        response_data = response.json()
    except json.JSONDecodeError as decode_error:
        LOGGER.warning("OpenRouter returned invalid JSON response: %s", decode_error)
        return None

    if not isinstance(response_data, dict):
        LOGGER.warning("OpenRouter returned unexpected response shape: %r", response_data)
        return None

    return _extract_message_content(typing.cast("JsonObject", response_data))


def _extract_text_block(content_block: object) -> str | None:
    if not isinstance(content_block, dict):
        return None
    text_value = content_block.get("text")
    return text_value if isinstance(text_value, str) else None


def _parse_json_object(completion_text: str | None) -> JsonObject | None:
    if not completion_text:
        return None

    try:
        parsed_response = json.loads(_remove_json_markdown_fence(completion_text))
    except json.JSONDecodeError:
        LOGGER.warning("OpenRouter returned non-JSON completion: %r", completion_text[:300])
        return None

    return typing.cast("JsonObject", parsed_response) if isinstance(parsed_response, dict) else None


def _remove_json_markdown_fence(completion_text: str) -> str:
    stripped = completion_text.strip()
    if not stripped.startswith("```"):
        return stripped

    markdown_lines = stripped.splitlines()
    if markdown_lines and markdown_lines[0].startswith("```"):
        markdown_lines = markdown_lines[1:]
    if markdown_lines and markdown_lines[-1].strip() == "```":
        markdown_lines = markdown_lines[:-1]
    return "\n".join(markdown_lines).strip()


def _extract_category_value(response_data: JsonObject) -> str | None:
    direct_category = response_data.get("category")
    if isinstance(direct_category, str):
        return direct_category

    predicted_flags = response_data.get("predicted_red_flags")
    if not isinstance(predicted_flags, list) or not predicted_flags:
        return None

    first_flag = predicted_flags[0]
    if not isinstance(first_flag, dict):
        return None

    nested_category = first_flag.get("category")
    return nested_category if isinstance(nested_category, str) else None
