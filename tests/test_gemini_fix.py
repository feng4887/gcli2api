import asyncio

from src.converter.gemini_fix import (
    _ensure_empty_tool_schema_for_claude,
    get_thinking_settings,
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


def test_model_lists_include_gemini_3_1_pro_and_gemini_3_6_flash():
    models = set(get_available_models())

    assert "gemini-3.1-pro-preview" in models
    assert "gemini-3.1-pro-preview-medium" in models
    assert "gemini-3.6-flash" in models
    assert "gemini-3.6-flash-minimal" in models
    assert "gemini-3.6-flash" in VERTEX_MODELS


def test_gemini_3_thinking_levels_match_model_capabilities():
    assert get_thinking_settings("gemini-3.1-pro-preview-medium") == (None, "MEDIUM")
    assert get_thinking_settings("gemini-3.6-flash-minimal") == (None, "MINIMAL")


def test_gemini_3_6_normalization_removes_unsupported_request_fields(monkeypatch):
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "false")
    request = {
        "model": "gemini-3.6-flash",
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

