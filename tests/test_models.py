# ruff: noqa: PLR2004

import asyncio
import types

from app.models import LLMClient, build_dialogue_chunks, parse_red_flag_category


def make_messages(message_count: int):
    return [
        types.SimpleNamespace(
            role="user" if one_message_index % 2 else "support", content=f"message {one_message_index}"
        )
        for one_message_index in range(message_count)
    ]


def test_parse_red_flag_category_clean() -> None:
    assert parse_red_flag_category('{"category":"clean","confidence":0.81}') is None


def test_parse_red_flag_category_flagged() -> None:
    assert parse_red_flag_category('{"category":"identity_deception","confidence":0.92}') == "identity_deception"


def test_parse_red_flag_category_rejects_unknown_label() -> None:
    assert parse_red_flag_category('{"category":"suspicious"}') is None


def test_parse_red_flag_category_accepts_markdown_json() -> None:
    assert parse_red_flag_category('```json\n{"category":"scope_violation"}\n```') == "scope_violation"


def test_build_dialogue_chunks_short_overlap() -> None:
    dialogue_chunks = build_dialogue_chunks(make_messages(5))

    assert [(one_chunk.start_message_index + 1, one_chunk.end_message_index + 1) for one_chunk in dialogue_chunks] == [
        (1, 4),
        (2, 5),
    ]


def test_build_dialogue_chunks_long_overlap() -> None:
    dialogue_chunks = build_dialogue_chunks(make_messages(36))

    assert [(one_chunk.start_message_index + 1, one_chunk.end_message_index + 1) for one_chunk in dialogue_chunks] == [
        (1, 20),
        (18, 36),
    ]


def test_build_dialogue_chunks_more_than_forty_messages() -> None:
    assert len(build_dialogue_chunks(make_messages(41))) == 2


def test_conflicting_chunk_categories_use_full_history_resolver(monkeypatch) -> None:
    request_labels: list[str] = []
    client = LLMClient()

    async def fake_request_completion(_dialogue_text: str, *, request_label: str) -> str:
        request_labels.append(request_label)
        if "conflict resolver" in request_label:
            return '{"reasoning":"full history resolves conflict","category":"policy_manipulation"}'
        if "chunk 1" in request_label:
            return '{"reasoning":"first chunk","category":"transaction_coercion"}'
        return '{"reasoning":"later chunk","category":"policy_manipulation"}'

    monkeypatch.setattr(client, "request_completion_async", fake_request_completion)

    assert asyncio.run(client.process_messages_classification(make_messages(9))) == "policy_manipulation"
    assert any("conflict resolver" in one_request_label for one_request_label in request_labels)
