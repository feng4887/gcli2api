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


def test_model_lists_include_gemini_3_1_pro_and_current_flash_models():
    models = set(get_available_models())

    assert "gemini-3.1-pro-preview" in models
    assert "gemini-3.1-pro-preview-medium" in models
    assert "gemini-3.6-flash" in models
    assert "gemini-3.6-flash-minimal" in models
    assert "gemini-3.6-flash" in VERTEX_MODELS
    assert "gemini-3.7-flash" in models
    assert "gemini-3.7-flash-high" in models
    assert "gemini-3.7-flash-medium" in models
    assert "gemini-3.7-flash-low" in models
    assert "gemini-3.7-flash-minimal" not in models
    assert "gemini-3.7-flash" in VERTEX_MODELS
    assert "gemini-3.8-flash" in models
    assert "gemini-3.8-flash-high" in models
    assert "gemini-3.8-flash-medium" in models
    assert "gemini-3.8-flash-low" in models
    assert "gemini-3.8-flash-minimal" not in models
    assert "gemini-3.8-flash" in VERTEX_MODELS


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

