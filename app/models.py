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

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_PROMPT_PATH = pathlib.Path(__file__).with_name("prompts") / "red_flag_classifier.md"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 16000
DEFAULT_BUDGET_TOKENS = 10000
DEFAULT_THINKING_OUTPUT_BUFFER = 2000  # запас токенов на ответ поверх budget_tokens
DEFAULT_VERBOSITY = "high"
DEFAULT_REASONING_ENABLED = True
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_REASONING_EXCLUDE = False
DEFAULT_LOG_REQUEST = False
DEFAULT_LOG_FULL_RESPONSE = True

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
        self.verbosity = os.getenv("OPENROUTER_VERBOSITY", DEFAULT_VERBOSITY).strip()
        self.reasoning_enabled = _read_bool_env("OPENROUTER_REASONING_ENABLED", DEFAULT_REASONING_ENABLED)
        self.reasoning_effort = os.getenv("OPENROUTER_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()
        self.reasoning_exclude = _read_bool_env("OPENROUTER_REASONING_EXCLUDE", DEFAULT_REASONING_EXCLUDE)
        self.max_tokens = _read_int_env("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        self.budget_tokens = _read_int_env("OPENROUTER_BUDGET_TOKENS", DEFAULT_BUDGET_TOKENS)
        self.log_request = _read_bool_env("OPENROUTER_LOG_REQUEST", DEFAULT_LOG_REQUEST)
        self.log_full_response = _read_bool_env("OPENROUTER_LOG_FULL_RESPONSE", DEFAULT_LOG_FULL_RESPONSE)
        self.prompt_text = _load_prompt()

    def process_dialogue_classification(self, dialogue_text: str) -> str | None:
        return parse_red_flag_category(self.request_completion(dialogue_text))

    def request_completion(self, dialogue_text: str) -> str | None:
        if not self.api_key:
            return None

        # max_tokens должен превышать budget_tokens при включённом reasoning
        max_tokens = (
            max(self.max_tokens, self.budget_tokens + DEFAULT_THINKING_OUTPUT_BUFFER)
            if self.reasoning_enabled
            else self.max_tokens
        )

        request_payload: JsonObject = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt_text},
                {"role": "user", "content": f"Диалог для классификации:\n\n{dialogue_text}"},
            ],
            "max_tokens": max_tokens,
            "verbosity": self.verbosity,
        }

        if self.reasoning_enabled:
            if self.model.startswith("anthropic/"):
                # OpenRouter: reasoning.max_tokens → Anthropic budget_tokens
                request_payload["reasoning"] = {"max_tokens": self.budget_tokens}
            else:
                # OpenRouter unified reasoning для OpenAI o-серии и др.
                request_payload["reasoning"] = {
                    "effort": self.reasoning_effort,
                    "exclude": self.reasoning_exclude,
                }
        else:
            # Без reasoning — используем структурированный вывод
            request_payload["response_format"] = RED_FLAG_RESPONSE_FORMAT

        try:
            if self.log_request:
                _record_request_payload(request_payload)
            response = httpx.post(
                OPENROUTER_API_URL,
                headers=_build_openrouter_headers(self.api_key),
                json=request_payload,
                timeout=self.timeout_seconds,
            )
            if self.log_full_response:
                _record_response_body(response)
                _record_reasoning(response)
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
    llm_client = LLMClient()
    if llm_client.log_request or llm_client.log_full_response:
        _record_llm_config(llm_client)
    return llm_client


def _load_prompt() -> str:
    prompt_path = pathlib.Path(os.getenv("RED_FLAG_PROMPT_PATH", str(DEFAULT_PROMPT_PATH)))
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


def _read_int_env(variable_name: str, default_value: int) -> int:
    raw_value = os.getenv(variable_name, "").strip()
    if not raw_value:
        return default_value
    try:
        return int(raw_value)
    except ValueError:
        LOGGER.warning("Invalid %s=%r, using %s", variable_name, raw_value, default_value)
        return default_value


def _read_float_env(variable_name: str, default_value: float) -> float:
    raw_value = os.getenv(variable_name, "").strip()
    if not raw_value:
        return default_value
    try:
        return float(raw_value)
    except ValueError:
        LOGGER.warning("Invalid %s=%r, using %s", variable_name, raw_value, default_value)
        return default_value


def _read_bool_env(variable_name: str, default_value: bool) -> bool:
    raw_value = os.getenv(variable_name, "").strip().lower()
    if not raw_value:
        return default_value
    if raw_value in {"1", "true", "yes", "y", "on"}:
        return True
    if raw_value in {"0", "false", "no", "n", "off"}:
        return False
    LOGGER.warning("Invalid %s=%r, using %s", variable_name, raw_value, default_value)
    return default_value


def _record_llm_config(llm_client: LLMClient) -> None:
    LOGGER.warning(
        "LLM config: model=%s timeout=%s verbosity=%s "
        "reasoning_enabled=%s reasoning_effort=%s reasoning_exclude=%s "
        "max_tokens=%s budget_tokens=%s "
        "log_request=%s log_full_response=%s api_key_present=%s",
        llm_client.model,
        llm_client.timeout_seconds,
        llm_client.verbosity,
        llm_client.reasoning_enabled,
        llm_client.reasoning_effort,
        llm_client.reasoning_exclude,
        llm_client.max_tokens,
        llm_client.budget_tokens,
        llm_client.log_request,
        llm_client.log_full_response,
        bool(llm_client.api_key),
    )


def _record_request_payload(request_payload: JsonObject) -> None:
    sanitized = dict(request_payload)
    messages = sanitized.pop("messages", None)
    if isinstance(messages, list):
        sanitized["message_count"] = len(messages)
    LOGGER.warning(
        "OpenRouter request:\n%s",
        json.dumps(sanitized, ensure_ascii=False, indent=2),
    )


def _record_response_body(response: httpx.Response) -> None:
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        response_body = response.text
    else:
        response_body = json.dumps(response_data, ensure_ascii=False, indent=2)
    LOGGER.warning("OpenRouter response status=%s body:\n%s", response.status_code, response_body)


def _record_reasoning(response: httpx.Response) -> None:
    """Логирует reasoning/thinking отдельным блоком для удобного чтения."""
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        return
    if not isinstance(response_data, dict):
        return

    response_choices = response_data.get("choices")
    if not isinstance(response_choices, list) or not response_choices:
        return
    response_message = response_choices[0].get("message", {})
    if not isinstance(response_message, dict):
        return

    # OpenRouter кладёт reasoning в message.reasoning (строка)
    reasoning_text = response_message.get("reasoning")
    if isinstance(reasoning_text, str) and reasoning_text.strip():
        LOGGER.warning("=== reasoning ===\n%s\n=== end reasoning ===", reasoning_text)
        return

    # Либо в content-блоках типа "thinking" (нативный Anthropic формат)
    message_content = response_message.get("content")
    if isinstance(message_content, list):
        for one_content_block in message_content:
            if isinstance(one_content_block, dict) and one_content_block.get("type") == "thinking":
                thinking_text = one_content_block.get("thinking", "")
                if thinking_text:
                    LOGGER.warning("=== reasoning ===\n%s\n=== end reasoning ===", thinking_text)
                    return

    LOGGER.warning("=== reasoning: (пусто) ===")


def _extract_response_content(response: httpx.Response) -> str | None:
    try:
        response_data = response.json()
    except json.JSONDecodeError as decode_error:
        LOGGER.warning("OpenRouter returned invalid JSON response: %s", decode_error)
        return None

    if not isinstance(response_data, dict):
        LOGGER.warning("OpenRouter returned unexpected response shape: %r", response_data)
        return None

    response_choices = response_data.get("choices")
    if not isinstance(response_choices, list) or not response_choices:
        return None
    response_message = response_choices[0].get("message")
    if not isinstance(response_message, dict):
        return None

    return _extract_message_content(response_message.get("content"))


def _extract_message_content(message_content: object) -> str | None:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        return (
            "\n".join(
                one_content_block.get("text", "")
                for one_content_block in message_content
                if (
                    isinstance(one_content_block, dict)
                    and one_content_block.get("type") == "text"
                    and one_content_block.get("text")
                )
            )
            or None
        )
    return None


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
    content_lines = stripped.splitlines()
    if content_lines and content_lines[0].startswith("```"):
        content_lines = content_lines[1:]
    if content_lines and content_lines[-1].strip() == "```":
        content_lines = content_lines[:-1]
    return "\n".join(content_lines).strip()


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
