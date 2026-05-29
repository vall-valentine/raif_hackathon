from app.models import parse_red_flag_category


def test_parse_red_flag_category_clean() -> None:
    assert parse_red_flag_category('{"category":"clean","confidence":0.81}') is None


def test_parse_red_flag_category_flagged() -> None:
    assert parse_red_flag_category('{"category":"identity_deception","confidence":0.92}') == "identity_deception"


def test_parse_red_flag_category_rejects_unknown_label() -> None:
    assert parse_red_flag_category('{"category":"suspicious"}') is None


def test_parse_red_flag_category_accepts_markdown_json() -> None:
    assert parse_red_flag_category('```json\n{"category":"scope_violation"}\n```') == "scope_violation"
