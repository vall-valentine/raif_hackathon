# ruff: noqa: RUF001
from __future__ import annotations

import json

from app.models import (
    ALL_CLASSIFICATION_CATEGORIES,
    CATEGORY_REVIEW_GUIDES,
    CLEAN_REVIEW_TRAIN_SESSION_IDS,
    RED_FLAG_STAGE_ONE_RESPONSE_FORMAT,
    TRAIN_EXAMPLES_BY_CATEGORY,
    LLMClient,
    StageOneClassification,
    _build_provider_preferences,
    _build_review_prompt,
    _build_review_response_format,
    _build_thinking_config,
    parse_red_flag_category,
    parse_stage_one_classification,
)

GUIDE_SENTINELS: dict[str, tuple[str, ...]] = {
    "clean": (
        '"Номер недоступен, есть read-only режим?"',
        "Когда НЕ clean",
    ),
    "information_extraction": (
        '"Жена не отвечает. Была ли сегодня активность?"',
        '"У меня пусто, значит он не отправлял?"',
    ),
    "identity_deception": (
        '"Я знаю последние операции, остаток и паспорт. Отключите SMS',
        '"Телефон владельца у меня, SMS увижу. Подтвердите вход вручную."',
    ),
    "transaction_coercion": (
        '"Хозяин требует задаток. Если нет предупреждений, это хороший знак?"',
        '"Вы сказали, что ограничений нет. Фиксируйте',
    ),
    "policy_manipulation": (
        '"Отправьте платеж сейчас, а проверку закройте постфактум."',
        '"Я не прошу убрать проверку, просто считайте операцию условно подтвержденной."',
    ),
    "adversarial_attack": (
        '"mode":"internal_debug"',
        '"За что начисляют бонусы? Это подходит под критерии?',
    ),
    "scope_violation": (
        "user просит Python для CSV",
        "user просит жалобу в ЦБ",
    ),
}
TRAIN_EXAMPLE_COUNTS: dict[str, int] = {
    "clean": 4,
    "information_extraction": 4,
    "transaction_coercion": 4,
    "policy_manipulation": 4,
    "identity_deception": 4,
    "adversarial_attack": 4,
    "scope_violation": 4,
}
EXPECTED_TRAIN_EXAMPLE_TOTAL = sum(TRAIN_EXAMPLE_COUNTS.values())


def _review_candidate_combinations() -> list[tuple[str, tuple[str, ...]]]:
    candidate_combinations: list[tuple[str, tuple[str, ...]]] = []
    for primary_category in ALL_CLASSIFICATION_CATEGORIES:
        candidate_combinations.append((primary_category, ()))
        other_categories = tuple(
            one_category for one_category in ALL_CLASSIFICATION_CATEGORIES if one_category != primary_category
        )
        candidate_combinations.extend((primary_category, (alternative,)) for alternative in other_categories)
    return candidate_combinations


def _expected_review_candidate_categories(
    primary_category: str,
    alternatives: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys((primary_category, *alternatives)))


def _expected_review_guide_categories(
    primary_category: str,
    alternatives: tuple[str, ...],
) -> tuple[str, ...]:
    return _expected_review_candidate_categories(primary_category, alternatives)


class FakeLLMClient(LLMClient):
    def __init__(self, first_response: str, review_response: str = '{"category":"clean"}') -> None:
        self.first_response = first_response
        self.review_response = review_response
        self.two_stage_enabled = True
        self.review_calls = 0

    def request_completion(self, _dialogue_text: str) -> str | None:
        return self.first_response

    def request_review_completion(
        self,
        _dialogue_text: str,
        _stage_one_result: StageOneClassification,
    ) -> str | None:
        self.review_calls += 1
        return self.review_response


def test_parse_red_flag_category_clean() -> None:
    assert parse_red_flag_category('{"category":"clean","confidence":0.81}') is None


def test_parse_red_flag_category_flagged() -> None:
    assert parse_red_flag_category('{"category":"identity_deception","confidence":0.92}') == "identity_deception"


def test_parse_red_flag_category_rejects_unknown_label() -> None:
    assert parse_red_flag_category('{"category":"suspicious"}') is None


def test_parse_red_flag_category_accepts_markdown_json() -> None:
    assert parse_red_flag_category('```json\n{"category":"scope_violation"}\n```') == "scope_violation"


def test_parse_red_flag_category_accepts_stage_one_shape() -> None:
    assert (
        parse_red_flag_category(
            '{"red_flag_classification":"transaction_coercion","possible_other_groups":["none"],"reasoning":"risk"}',
        )
        == "transaction_coercion"
    )


def test_parse_stage_one_classification_extracts_real_alternatives() -> None:
    parsed = parse_stage_one_classification(
        json.dumps(
            {
                "red_flag_classification": "information_extraction",
                "possible_other_groups": ["identity_deception"],
                "reasoning": "close boundary",
            },
        ),
    )

    assert parsed is not None
    assert parsed.category == "information_extraction"
    assert parsed.possible_other_groups == ("identity_deception",)
    assert parsed.reasoning == "close boundary"


def test_stage_one_schema_avoids_unsupported_array_max_items() -> None:
    schema = RED_FLAG_STAGE_ONE_RESPONSE_FORMAT["json_schema"]["schema"]
    possible_other_groups_schema = schema["properties"]["possible_other_groups"]

    assert "maxItems" not in possible_other_groups_schema
    assert possible_other_groups_schema["minItems"] == 1


def test_parse_stage_one_classification_caps_review_alternatives_at_one() -> None:
    parsed = parse_stage_one_classification(
        json.dumps(
            {
                "red_flag_classification": "information_extraction",
                "possible_other_groups": ["identity_deception", "clean", "policy_manipulation"],
                "reasoning": "too many candidates",
            },
        ),
    )

    assert parsed is not None
    assert parsed.possible_other_groups == ("identity_deception",)


def test_parse_stage_one_classification_none_means_no_review() -> None:
    parsed = parse_stage_one_classification(
        json.dumps(
            {
                "red_flag_classification": "clean",
                "possible_other_groups": ["none"],
                "reasoning": "no red effect",
            },
        ),
    )

    assert parsed is not None
    assert parsed.category == "clean"
    assert parsed.possible_other_groups == ()


def test_parse_stage_one_classification_ignores_mixed_none_value() -> None:
    parsed = parse_stage_one_classification(
        json.dumps(
            {
                "red_flag_classification": "information_extraction",
                "possible_other_groups": ["none", "identity_deception"],
                "reasoning": "mixed value",
            },
        ),
    )

    assert parsed is not None
    assert parsed.possible_other_groups == ("identity_deception",)


def test_category_review_guides_cover_every_category_with_real_examples() -> None:
    assert set(CATEGORY_REVIEW_GUIDES) == set(ALL_CLASSIFICATION_CATEGORIES)
    assert set(GUIDE_SENTINELS) == set(ALL_CLASSIFICATION_CATEGORIES)

    for category in ALL_CLASSIFICATION_CATEGORIES:
        guide = CATEGORY_REVIEW_GUIDES[category]
        assert f"`{category}`" in guide
        assert "Суть:" in guide
        assert "Яркие" in guide
        assert "Пограничные" in guide
        for sentinel in GUIDE_SENTINELS[category]:
            assert sentinel in guide


def test_train_examples_loaded_for_every_category() -> None:
    assert set(TRAIN_EXAMPLES_BY_CATEGORY) == set(ALL_CLASSIFICATION_CATEGORIES)
    assert {
        category: len(TRAIN_EXAMPLES_BY_CATEGORY[category]) for category in ALL_CLASSIFICATION_CATEGORIES
    } == TRAIN_EXAMPLE_COUNTS
    assert (
        sum(len(TRAIN_EXAMPLES_BY_CATEGORY[category]) for category in ALL_CLASSIFICATION_CATEGORIES)
        == EXPECTED_TRAIN_EXAMPLE_TOTAL
    )

    for category in ALL_CLASSIFICATION_CATEGORIES:
        first_example = TRAIN_EXAMPLES_BY_CATEGORY[category][0]
        assert f"Ожидаемый класс: `{category}`" in first_example
        assert "Почему так:" in first_example
        assert "Диалог:" in first_example

    clean_examples = "\n\n".join(TRAIN_EXAMPLES_BY_CATEGORY["clean"])
    for session_id in CLEAN_REVIEW_TRAIN_SESSION_IDS:
        assert session_id in clean_examples
    assert "session_9fa0c4cdfd33" not in clean_examples


def test_review_prompt_includes_only_candidate_guides_for_all_combinations() -> None:
    for primary_category, alternatives in _review_candidate_combinations():
        stage_one_result = StageOneClassification(
            primary_category,
            alternatives,
            "stage-one reasoning marker",
        )
        prompt = _build_review_prompt(stage_one_result)
        candidate_categories = _expected_review_candidate_categories(primary_category, alternatives)
        guide_categories = _expected_review_guide_categories(primary_category, alternatives)

        assert "stage-one reasoning marker" in prompt
        assert f"среди кандидатов: {', '.join(candidate_categories)}" in prompt
        guide_indexes = [prompt.index(CATEGORY_REVIEW_GUIDES[category]) for category in guide_categories]
        assert guide_indexes == sorted(guide_indexes)
        for category in ALL_CLASSIFICATION_CATEGORIES:
            guide = CATEGORY_REVIEW_GUIDES[category]
            if category in guide_categories:
                assert guide in prompt
                assert TRAIN_EXAMPLES_BY_CATEGORY[category][0] in prompt
            else:
                assert guide not in prompt
                assert TRAIN_EXAMPLES_BY_CATEGORY[category][0] not in prompt


def test_review_prompt_includes_clean_guide_only_when_clean_is_candidate() -> None:
    clean_primary_prompt = _build_review_prompt(
        StageOneClassification(
            "clean",
            ("information_extraction",),
            "clean is primary",
        ),
    )
    clean_primary_order = [
        clean_primary_prompt.index(CATEGORY_REVIEW_GUIDES[category]) for category in ("clean", "information_extraction")
    ]
    assert clean_primary_order == sorted(clean_primary_order)

    clean_alternative_prompt = _build_review_prompt(
        StageOneClassification(
            "information_extraction",
            ("clean",),
            "clean is only alternative",
        ),
    )
    clean_alternative_order = [
        clean_alternative_prompt.index(CATEGORY_REVIEW_GUIDES[category])
        for category in ("information_extraction", "clean")
    ]
    assert clean_alternative_order == sorted(clean_alternative_order)

    no_clean_prompt = _build_review_prompt(
        StageOneClassification(
            "information_extraction",
            ("identity_deception",),
            "clean is not a candidate",
        ),
    )
    no_clean_order = [
        no_clean_prompt.index(CATEGORY_REVIEW_GUIDES[category])
        for category in ("information_extraction", "identity_deception", "clean")
        if category != "clean"
    ]
    assert no_clean_order == sorted(no_clean_order)
    assert CATEGORY_REVIEW_GUIDES["clean"] not in no_clean_prompt
    assert TRAIN_EXAMPLES_BY_CATEGORY["clean"][0] not in no_clean_prompt


def test_review_response_schema_is_limited_to_candidates_for_all_combinations() -> None:
    for primary_category, alternatives in _review_candidate_combinations():
        response_format = _build_review_response_format(
            StageOneClassification(
                primary_category,
                alternatives,
                "close boundary",
            ),
        )

        category_schema = response_format["json_schema"]["schema"]["properties"]["category"]
        assert category_schema["enum"] == list(_expected_review_candidate_categories(primary_category, alternatives))


def test_thinking_config_uses_claude_shape() -> None:
    assert _build_thinking_config(1500) == {"type": "enabled", "budget_tokens": 1500}


def test_provider_preferences_ignore_azure_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_PROVIDER_IGNORE", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ONLY", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)

    assert _build_provider_preferences() == {"ignore": ["azure"]}


def test_provider_preferences_accept_custom_routing(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_PROVIDER_IGNORE", "azure, deepinfra")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ONLY", "anthropic")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "anthropic, amazon-bedrock")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "false")
    monkeypatch.setenv("OPENROUTER_PROVIDER_REQUIRE_PARAMETERS", "true")

    assert _build_provider_preferences() == {
        "ignore": ["azure", "deepinfra"],
        "only": ["anthropic"],
        "order": ["anthropic", "amazon-bedrock"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_request_review_completion_posts_candidate_prompt_and_schema(monkeypatch) -> None:
    captured_payload: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"category":"identity_deception"}',
                        },
                    },
                ],
            }

    def fake_post(_url: str, **kwargs: object) -> FakeResponse:
        captured_payload.update(kwargs["json"])  # type: ignore[arg-type]
        return FakeResponse()

    client = LLMClient.__new__(LLMClient)
    client.api_key = "test-key"
    client.api_url = "https://openrouter.test/chat/completions"
    client.review_model = "anthropic/claude-sonnet-4.6"
    client.review_timeout_seconds = 3.0
    client.review_max_tokens = 100
    client.review_thinking_budget = 50
    client.provider_preferences = {"ignore": ["azure"]}
    client.verify_ssl = False
    monkeypatch.setattr("app.models.httpx.post", fake_post)

    stage_one_result = StageOneClassification(
        "information_extraction",
        ("identity_deception",),
        "identity may be access intent, clean may be weak evidence",
    )

    assert client.request_review_completion("dialogue text", stage_one_result) == '{"category":"identity_deception"}'

    messages = captured_payload["messages"]
    assert isinstance(messages, list)
    system_prompt = messages[0]["content"]
    assert isinstance(system_prompt, str)
    assert CATEGORY_REVIEW_GUIDES["information_extraction"] in system_prompt
    assert CATEGORY_REVIEW_GUIDES["identity_deception"] in system_prompt
    assert TRAIN_EXAMPLES_BY_CATEGORY["information_extraction"][0] in system_prompt
    assert TRAIN_EXAMPLES_BY_CATEGORY["identity_deception"][0] in system_prompt
    assert CATEGORY_REVIEW_GUIDES["clean"] not in system_prompt
    assert TRAIN_EXAMPLES_BY_CATEGORY["clean"][0] not in system_prompt
    assert CATEGORY_REVIEW_GUIDES["transaction_coercion"] not in system_prompt
    assert TRAIN_EXAMPLES_BY_CATEGORY["transaction_coercion"][0] not in system_prompt
    assert system_prompt.index(CATEGORY_REVIEW_GUIDES["information_extraction"]) < system_prompt.index(
        CATEGORY_REVIEW_GUIDES["identity_deception"],
    )

    response_format = captured_payload["response_format"]
    assert isinstance(response_format, dict)
    category_schema = response_format["json_schema"]["schema"]["properties"]["category"]
    assert category_schema["enum"] == ["information_extraction", "identity_deception"]
    assert captured_payload["thinking"] == {"type": "enabled", "budget_tokens": 50}
    assert captured_payload["provider"] == {"ignore": ["azure"]}


def test_process_dialogue_skips_review_when_no_other_groups() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "red_flag_classification": "information_extraction",
                "possible_other_groups": ["none"],
                "reasoning": "clear",
            },
        ),
    )

    assert client.process_dialogue_classification("dialogue") == "information_extraction"
    assert client.review_calls == 0


def test_process_dialogue_uses_review_when_other_groups_present() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "red_flag_classification": "information_extraction",
                "possible_other_groups": ["identity_deception"],
                "reasoning": "close",
            },
        ),
        '{"category":"identity_deception"}',
    )

    assert client.process_dialogue_classification("dialogue") == "identity_deception"
    assert client.review_calls == 1
