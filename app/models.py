"""OpenRouter baseline for session-level red-flag classification."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pathlib
import typing

import httpx

LOGGER = logging.getLogger("uvicorn.error")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_PROMPT_PATH = pathlib.Path(__file__).with_name("prompts") / "red_flag_classifier.md"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 2000
DEFAULT_THINKING_BUDGET = 1500
DEFAULT_VERIFY_SSL = False
MAX_LOG_CHARS = 2000
MIN_QUOTED_DOTENV_VALUE_LENGTH = 2
MIN_CHUNK_MESSAGES = 4
CHUNK_OVERLAP_MESSAGES = 3
VERY_SHORT_DIALOGUE_MESSAGES = 4

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
DialogueMessageLike = typing.Any


@typing.final
class DialogueChunk(typing.NamedTuple):
    chunk_index: int
    start_message_index: int
    end_message_index: int
    chunk_text: str


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
        self.api_key = _read_env_variable("OPENROUTER_API_KEY")
        self.model = _read_env_variable("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.timeout_seconds = _read_float_env("OPENROUTER_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        self.verify_ssl = _read_bool_env("OPENROUTER_VERIFY_SSL", DEFAULT_VERIFY_SSL)
        self.prompt_text = _load_prompt()

    async def process_messages_classification(self, messages: typing.Sequence[DialogueMessageLike]) -> str | None:
        dialogue_chunks = build_dialogue_chunks(messages)
        if not dialogue_chunks:
            LOGGER.info("Final red-flag category: %s", CLEAN_LABEL)
            return None

        LOGGER.info("Chunked dialogue: messages=%s chunks=%s", len(messages), len(dialogue_chunks))
        chunk_completions = await self.request_chunk_completions(dialogue_chunks)
        chunk_categories = [
            one_category
            for one_category in (parse_red_flag_category(one_completion) for one_completion in chunk_completions)
            if one_category
        ]
        final_category = await self.resolve_chunk_categories(messages, chunk_categories)
        LOGGER.info("Final red-flag category: %s", final_category or CLEAN_LABEL)
        return final_category

    async def request_chunk_completions(self, dialogue_chunks: typing.Sequence[DialogueChunk]) -> list[str | None]:
        return list(
            await asyncio.gather(
                *(
                    self.request_completion_async(
                        one_dialogue_chunk.chunk_text,
                        request_label=(
                            f"chunk {one_dialogue_chunk.chunk_index + 1}/{len(dialogue_chunks)} "
                            f"messages={one_dialogue_chunk.start_message_index + 1}-"
                            f"{one_dialogue_chunk.end_message_index + 1}"
                        ),
                    )
                    for one_dialogue_chunk in dialogue_chunks
                ),
            ),
        )

    async def resolve_chunk_categories(
        self,
        messages: typing.Sequence[DialogueMessageLike],
        chunk_categories: typing.Sequence[str],
    ) -> str | None:
        if not chunk_categories:
            return None

        unique_categories = get_unique_categories(chunk_categories)
        if len(unique_categories) == 1:
            return unique_categories[0]

        LOGGER.info("Conflicting chunk categories: %s", ", ".join(unique_categories))
        resolver_text = format_conflict_resolution_dialogue(messages, unique_categories)
        resolver_completion = await self.request_completion_async(
            resolver_text,
            request_label=f"conflict resolver candidates={','.join(unique_categories)}",
        )
        resolver_category = parse_red_flag_category(resolver_completion)
        if resolver_category in unique_categories:
            return resolver_category

        fallback_category = choose_first_frequent_category(chunk_categories)
        LOGGER.warning(
            "Conflict resolver returned %s; fallback category: %s",
            resolver_category or CLEAN_LABEL,
            fallback_category,
        )
        return fallback_category

    async def request_completion_async(self, dialogue_text: str, *, request_label: str) -> str | None:
        if not self.api_key:
            LOGGER.warning("OPENROUTER_API_KEY is not configured; returning clean fallback")
            return None

        LOGGER.info(
            "OpenRouter async request: %s model=%s dialogue_chars=%s structured_output=json_schema verify_ssl=%s",
            request_label,
            self.model,
            len(dialogue_text),
            self.verify_ssl,
        )
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                trust_env=False,
                verify=self.verify_ssl,
            ) as async_client:
                response = await async_client.post(
                    OPENROUTER_API_URL,
                    headers=_build_openrouter_headers(self.api_key),
                    json=build_openrouter_payload(self.model, self.prompt_text, dialogue_text),
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as request_error:
            LOGGER.warning(
                "OpenRouter HTTP %s response for %s: %s",
                request_error.response.status_code,
                request_label,
                _format_text_for_log(request_error.response.text),
            )
            return None
        except (httpx.HTTPError, OSError) as request_error:
            LOGGER.warning("OpenRouter request failed for %s: %s", request_label, request_error)
            return None

        return _extract_response_content(response)


def build_dialogue_chunks(messages: typing.Sequence[DialogueMessageLike]) -> list[DialogueChunk]:
    message_count = len(messages)
    if message_count == 0:
        return []

    chunk_ranges = build_chunk_ranges(message_count)
    return [
        DialogueChunk(
            chunk_index=one_chunk_index,
            start_message_index=one_start_index,
            end_message_index=one_stop_index - 1,
            chunk_text=format_dialogue_slice(messages, one_start_index, one_stop_index),
        )
        for one_chunk_index, (one_start_index, one_stop_index) in enumerate(chunk_ranges)
    ]


def choose_chunk_count(message_count: int) -> int:
    if message_count <= VERY_SHORT_DIALOGUE_MESSAGES:
        return 1
    return 2


def build_chunk_ranges(message_count: int) -> list[tuple[int, int]]:
    if choose_chunk_count(message_count) == 1:
        return [(0, message_count)]

    first_stop_index = min(
        message_count,
        max(MIN_CHUNK_MESSAGES, math.ceil((message_count + CHUNK_OVERLAP_MESSAGES) / 2)),
    )
    return [(0, first_stop_index), (max(0, first_stop_index - CHUNK_OVERLAP_MESSAGES), message_count)]


def format_dialogue_slice(
    messages: typing.Sequence[DialogueMessageLike],
    start_message_index: int,
    stop_message_index: int,
) -> str:
    return "\n".join(
        f"{one_message_index + 1}. {one_message.role}: {one_message.content}"
        for one_message_index, one_message in enumerate(
            messages[start_message_index:stop_message_index],
            start=start_message_index,
        )
    )


def format_conflict_resolution_dialogue(
    messages: typing.Sequence[DialogueMessageLike],
    unique_categories: typing.Sequence[str],
) -> str:
    return (
        "Уточняющая классификация полной истории.\n"
        f"chunk-классификация выбрала разные red-flag категории: {', '.join(unique_categories)}.\n"
        "Нужно выбрать одну итоговую red-flag категорию для всей сессии. "
        "Проанализируй полную историю и выбери главный риск из категорий-кандидатов.\n\n"
        f"Полная история:\n{format_dialogue_slice(messages, 0, len(messages))}"
    )


def build_openrouter_payload(model_name: str, prompt_text: str, dialogue_text: str) -> JsonObject:
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Диалог для классификации:\n\n{dialogue_text}"},
        ],
        "max_tokens": DEFAULT_MAX_TOKENS,
        "reasoning": {
            "max_tokens": DEFAULT_THINKING_BUDGET,
            "exclude": True,
        },
        "response_format": RED_FLAG_RESPONSE_FORMAT,
    }


def get_unique_categories(categories: typing.Sequence[str]) -> list[str]:
    unique_categories: list[str] = []
    for one_category in categories:
        if one_category not in unique_categories:
            unique_categories.append(one_category)
    return unique_categories


def choose_first_frequent_category(categories: typing.Sequence[str]) -> str:
    category_counts: dict[str, int] = {}
    for one_category in categories:
        category_counts[one_category] = category_counts.get(one_category, 0) + 1

    best_category = categories[0]
    best_category_count = category_counts[best_category]
    for one_category, one_category_count in category_counts.items():
        if one_category_count > best_category_count:
            best_category = one_category
            best_category_count = one_category_count
    return best_category


def parse_red_flag_category(completion_text: str | None) -> str | None:
    response_data = _parse_json_object(completion_text)
    if response_data is None:
        return None

    category = _extract_category_value(response_data)
    if category is None:
        return None

    normalized_category = category.strip().lower()
    if normalized_category in {CLEAN_LABEL, "", "none", "null"}:
        LOGGER.info("OpenRouter parsed category: clean")
        return None
    if normalized_category in RED_FLAG_CATEGORIES:
        LOGGER.info("OpenRouter parsed category: %s", normalized_category)
        return normalized_category

    LOGGER.warning("OpenRouter returned unsupported red-flag category: %s", category)
    return None


async def process_risk_detection(
    llm_client: LLMClient,
    messages: typing.Sequence[DialogueMessageLike],
) -> dict[str, str] | None:
    """Classify one dialogue and return evaluator-compatible output."""
    category = await llm_client.process_messages_classification(messages)
    if category is None:
        return None
    return {"category": category}


def load_llm() -> LLMClient:
    """Create an OpenRouter client at application startup."""
    return LLMClient()


def _load_prompt() -> str:
    prompt_path = pathlib.Path(_read_env_variable("RED_FLAG_PROMPT_PATH", str(DEFAULT_PROMPT_PATH)))
    try:
        return prompt_path.read_text(encoding="utf-8")
    except OSError as prompt_error:
        LOGGER.warning("Could not read prompt file %s: %s", prompt_path, prompt_error)
        return "Classify the dialogue as clean or one red-flag category. Return JSON with key category."


def _read_env_variable(variable_name: str, default_value: str = "") -> str:
    env_value = os.getenv(variable_name, "").strip()
    if env_value:
        return env_value
    return _read_dotenv_value(variable_name) or default_value


def _read_dotenv_value(variable_name: str) -> str:
    try:
        dotenv_lines = DEFAULT_DOTENV_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    variable_prefix = f"{variable_name}="
    for one_dotenv_line in dotenv_lines:
        normalized_line = one_dotenv_line.strip()
        if not normalized_line or normalized_line.startswith("#") or not normalized_line.startswith(variable_prefix):
            continue
        return _remove_dotenv_quotes(normalized_line.removeprefix(variable_prefix).strip())
    return ""


def _remove_dotenv_quotes(raw_value: str) -> str:
    if len(raw_value) < MIN_QUOTED_DOTENV_VALUE_LENGTH:
        return raw_value
    if raw_value[0] == raw_value[-1] and raw_value[0] in {'"', "'"}:
        return raw_value[1:-1]
    return raw_value


def _build_openrouter_headers(openrouter_api_key: str) -> dict[str, str]:
    request_headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }
    referer_header = _read_env_variable("OPENROUTER_SITE_URL")
    app_name = _read_env_variable("OPENROUTER_APP_NAME", "Red Flag Detector")
    if referer_header:
        request_headers["HTTP-Referer"] = referer_header
    if app_name:
        request_headers["X-Title"] = app_name
    return request_headers


def _read_float_env(variable_name: str, default_value: float) -> float:
    raw_value = _read_env_variable(variable_name)
    if not raw_value:
        return default_value
    try:
        return float(raw_value)
    except ValueError:
        LOGGER.warning("Invalid %s=%r, using %s", variable_name, raw_value, default_value)
        return default_value


def _read_bool_env(variable_name: str, default_value: bool) -> bool:
    raw_value = _read_env_variable(variable_name).lower()
    if not raw_value:
        return default_value
    if raw_value in {"1", "true", "yes", "y", "on"}:
        return True
    if raw_value in {"0", "false", "no", "n", "off"}:
        return False
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

    completion_text = _extract_message_content(typing.cast("JsonObject", response_data))
    if completion_text:
        LOGGER.info("OpenRouter raw message content: %s", _format_text_for_log(completion_text))
    else:
        LOGGER.warning("OpenRouter response has no message content: %s", _format_text_for_log(str(response_data)))
    return completion_text


def _format_text_for_log(raw_text: str) -> str:
    if len(raw_text) <= MAX_LOG_CHARS:
        return raw_text
    return f"{raw_text[:MAX_LOG_CHARS]}...<truncated>"


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
