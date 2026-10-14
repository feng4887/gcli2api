import asyncio

from src.converter.gemini_fix import (
    _ensure_empty_tool_schema_for_claude,
    get_thinking_settings,
    map_antigravity_gemini_model,
    normalize_gemini_request,
)
from src.router.vertex.model_list import VERTEX_MODELS
from src.utils import get_available_models


def test_antigravity_claude_tools_keep_schema_in_parameters():
    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "test_tool",
                    "description": "A test tool.",
                    "parametersJsonSchema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                }
            ]
        }
    ]

    result = _ensure_empty_tool_schema_for_claude(tools, "claude-opus-4-6-thinking", "antigravity")
    declaration = result[0]["functionDeclarations"][0]

    assert declaration["parameters"]["type"] == "object"
    assert "parametersJsonSchema" not in declaration


def test_model_lists_track_current_backend_offering():
    """对外广告的模型必须与后端当前目录一致（2026-10-14 实测集合）。"""
    models = set(get_available_models())

    for offered in (
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-image",
        "gemini-3.8-flash",
    ):
        assert offered in models
        assert offered in VERTEX_MODELS

    assert "gemini-3.8-flash-high" in models
    assert "gemini-3.8-flash-medium" in models
    assert "gemini-3.8-flash-low" in models
    assert "gemini-3.8-flash-minimal" not in models

    for retired in (
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3-flash-preview",
        "gemini-4-argon",
    ):
        assert retired not in models, f"{retired} 已从后端下线，不应再对外广告"
        assert retired not in VERTEX_MODELS



def test_antigravity_downgrades_tier_aliases_when_backend_advertises_tiered_only():
    """后端只广告 tiered 档位时，固定档位与裸名必须收敛到 tiered，而不是 404。"""
    from src.converter import gemini_fix

    gemini_fix.set_antigravity_available_model_ids(
        {"gemini-3.8-flash-tiered", "gemini-3-flash", "claude-opus-4-6-thinking"}
    )
    try:
        assert (
            map_antigravity_gemini_model("gemini-3.8-flash-high", "HIGH", None)
            == "gemini-3.8-flash-tiered"
        )
        assert (
            map_antigravity_gemini_model("gemini-3.8-flash", None, None)
            == "gemini-3.8-flash-tiered"
        )
        # 后端目录里没有的旧别名收敛到同族在线的模型
        assert (
            map_antigravity_gemini_model("gemini-3-flash-high", "HIGH", None)
            == "gemini-3-flash"
        )
        assert (
            map_antigravity_gemini_model("gemini-3-flash-preview", None, None)
            == "gemini-3-flash"
        )
        # 跨模型不做替换：后端没有的 ID 原样返回（由后端报 404）
        assert map_antigravity_gemini_model("gemini-4-argon", None, None) == "gemini-4-argon"
    finally:
        gemini_fix.set_antigravity_available_model_ids(set())


def test_gemini_3_thinking_levels_match_model_capabilities():

    assert get_thinking_settings("gemini-3.1-pro-preview-medium") == (None, "MEDIUM")
    assert get_thinking_settings("gemini-3.6-flash-minimal") == (None, "MINIMAL")
    assert get_thinking_settings("gemini-3.7-flash-high") == (None, "HIGH")
    assert get_thinking_settings("gemini-3.7-flash-medium") == (None, "MEDIUM")
    assert get_thinking_settings("gemini-3.7-flash-low") == (None, "LOW")
    assert get_thinking_settings("gemini-3.7-flash-minimal") == (None, "LOW")
    assert get_thinking_settings("gemini-3.8-flash-high") == (None, "HIGH")
    assert get_thinking_settings("gemini-3.8-flash-medium") == (None, "MEDIUM")
    assert get_thinking_settings("gemini-3.8-flash-low") == (None, "LOW")
    assert get_thinking_settings("gemini-3.8-flash-minimal") == (None, "LOW")


def test_antigravity_maps_current_gemini_flash_models_without_dead_branches():
    for base_model in ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"):
        assert (
            map_antigravity_gemini_model(f"{base_model}-high", "LOW", None)
            == f"{base_model}-high"
        )
        assert (
            map_antigravity_gemini_model(f"{base_model}-medium", "HIGH", None)
            == f"{base_model}-medium"
        )
        assert (
            map_antigravity_gemini_model(f"{base_model}-low", "HIGH", None)
            == f"{base_model}-low"
        )
        assert (
            map_antigravity_gemini_model(f"{base_model}-tiered", "HIGH", None)
            == f"{base_model}-tiered"
        )
        assert map_antigravity_gemini_model(base_model, "HIGH", None) == f"{base_model}-high"
        assert map_antigravity_gemini_model(base_model, "MEDIUM", None) == f"{base_model}-medium"
        assert map_antigravity_gemini_model(base_model, "LOW", None) == f"{base_model}-low"
        assert map_antigravity_gemini_model(base_model, None, None) == f"{base_model}-medium"


def test_antigravity_normalization_maps_gemini_3_8_and_applies_request_rules(monkeypatch):
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "false")
    request = {
        "model": "gemini-3.8-flash",
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": "HIGH"},
            "temperature": 0.7,
            "topP": 0.9,
            "topK": 40,
            "candidateCount": 2,
            "frequencyPenalty": 0.5,
            "presencePenalty": 0.5,
        },
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Prefilled response"}]},
        ],
    }

    result = asyncio.run(normalize_gemini_request(request, mode="antigravity"))

    assert result["model"] == "gemini-3.8-flash-high"
    assert result["generationConfig"] == {"maxOutputTokens": 64000}
    assert result["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]


def test_gemini_3_7_and_3_8_migrate_legacy_thinking_budget_to_supported_level(monkeypatch):
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "false")
    for model in ("gemini-3.7-flash", "gemini-3.8-flash"):
        request = {
            "model": model,
            "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}},
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }

        result = asyncio.run(normalize_gemini_request(request, mode="geminicli"))

        assert result["generationConfig"] == {
            "thinkingConfig": {"thinkingLevel": "LOW", "includeThoughts": False},
            "maxOutputTokens": 64000,
        }


def test_current_gemini_flash_normalization_removes_unsupported_request_fields(monkeypatch):
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "false")
    for model in ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"):
        request = {
            "model": model,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "topK": 40,
                "candidateCount": 2,
            },
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Prefilled response"}]},
            ],
        }

        result = asyncio.run(normalize_gemini_request(request, mode="geminicli"))

        assert result["generationConfig"] == {"maxOutputTokens": 64000}
        assert result["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]


def test_gemini_3_8_normalization_removes_penalty_fields(monkeypatch):
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "false")
    request = {
        "model": "gemini-3.8-flash",
        "generationConfig": {
            "frequencyPenalty": 0.5,
            "presencePenalty": 0.5,
        },
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
    }

    result = asyncio.run(normalize_gemini_request(request, mode="geminicli"))

    assert result["generationConfig"] == {"maxOutputTokens": 64000}

