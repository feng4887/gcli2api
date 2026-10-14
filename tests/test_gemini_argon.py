"""Verify Argon discovery and request compatibility without calling Google."""

import asyncio
import copy
import json

import pytest

from src.converter.gemini_fix import get_thinking_settings, normalize_gemini_request
from src.converter.openai2gemini import convert_openai_to_gemini_request
from src.router.antigravity import model_list as antigravity_models
from src.router.geminicli import model_list as cli_models
from src.router.vertex import model_list as vertex_models
from src.utils import get_available_models


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch):
    monkeypatch.setenv("COMPATIBILITY_MODE", "false")


def test_argon_feature_combinations_and_model_routes():
    models = get_available_models()
    assert len(models) == len(set(models))
    for prefix in ("", "假流式/", "流式抗截断/"):
        for level in ("", "-low", "-medium", "-high"):
            for search in ("", "-search"):
                assert f"{prefix}gemini-4-argon{level}{search}" in models
        assert f"{prefix}gemini-4-argon-minimal" not in models
    for module in (cli_models, vertex_models):
        openai_body = json.loads(asyncio.run(module.list_openai_models("test")).body)
        gemini_body = json.loads(asyncio.run(module.list_gemini_models("test")).body)
        assert "gemini-4-argon" in {entry["id"] for entry in openai_body["data"]}
        assert "models/gemini-4-argon" in {entry["name"] for entry in gemini_body["models"]}


@pytest.mark.parametrize("suffix,level", [
    ("", None), ("-low", "LOW"), ("-medium", "MEDIUM"), ("-high", "HIGH"),
    ("-minimal", "LOW"), ("-nothinking", "LOW"), ("-maxthinking", "HIGH"),
    ("-high-search", "HIGH"),
])
def test_argon_thinking_aliases(suffix, level):
    assert get_thinking_settings(f"gemini-4-argon{suffix}") == (None, level)


@pytest.mark.parametrize("mode", ["geminicli", "vertex", "antigravity"])
def test_argon_normalization_preserves_level_and_removes_unsupported_fields(mode, monkeypatch):
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "true")
    request = {
        "model": "gemini-4-argon-high",
        "cachedContent": "cachedContents/unsupported",
        "generationConfig": {
            "thinkingConfig": {"thinkingBudget": 0, "thinkingLevel": "MINIMAL"},
            "temperature": 0.7, "topP": 0.9, "topK": 40, "candidateCount": 2,
            "frequencyPenalty": 0.5, "presencePenalty": 0.5, "maxOutputTokens": 32768,
        },
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Prefilled"}]},
        ],
    }
    original = copy.deepcopy(request)
    result = asyncio.run(normalize_gemini_request(request, mode=mode))
    assert result["model"] == "gemini-4-argon"
    assert result["generationConfig"] == {
        "thinkingConfig": {"thinkingLevel": "HIGH", "includeThoughts": True},
        "maxOutputTokens": 32768,
    }
    assert result["contents"] == [original["contents"][0]]
    assert "cachedContent" not in result
    assert request == original


@pytest.mark.parametrize("mode", ["geminicli", "vertex", "antigravity"])
@pytest.mark.parametrize("thinking,level", [
    ({"thinkingBudget": 0}, "LOW"), ({"thinkingBudget": 8192}, "MEDIUM"),
    ({"thinkingBudget": 16000}, "HIGH"), ({"thinkingLevel": "MINIMAL"}, "LOW"),
    ({"thinkingLevel": "medium"}, "MEDIUM"),
])
def test_argon_migrates_legacy_thinking(mode, thinking, level, monkeypatch):
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "false")
    result = asyncio.run(normalize_gemini_request({
        "model": "gemini-4-argon",
        "generationConfig": {"thinkingConfig": thinking, "maxOutputTokens": 1000000},
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
    }, mode=mode))
    assert result["model"] == "gemini-4-argon"
    assert result["generationConfig"] == {
        "thinkingConfig": {"thinkingLevel": level, "includeThoughts": False},
        "maxOutputTokens": 64000,
    }


def test_argon_search_and_backend_default(monkeypatch):
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "false")
    result = asyncio.run(normalize_gemini_request({
        "model": "gemini-4-argon-search",
        "contents": [{"role": "user", "parts": [{"text": "Search"}]}],
    }))
    assert result["model"] == "gemini-4-argon"
    assert result["tools"] == [{"googleSearch": {}}]
    assert not result.get("generationConfig")


def test_argon_openai_conversion_and_dynamic_antigravity_discovery(monkeypatch):
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "false")
    converted = asyncio.run(convert_openai_to_gemini_request({
        "model": "gemini-4-argon-medium-search",
        "messages": [{"role": "user", "content": "Hello"}],
        "n": 1, "temperature": 1, "frequency_penalty": 0, "max_tokens": 4096,
    }))
    # The OpenAI router supplies the model after the format conversion.
    converted["model"] = "gemini-4-argon-medium-search"
    result = asyncio.run(normalize_gemini_request(converted))
    assert result["model"] == "gemini-4-argon"
    assert result["generationConfig"] == {
        "thinkingConfig": {"thinkingLevel": "MEDIUM", "includeThoughts": False},
        "maxOutputTokens": 4096,
    }
    assert result["tools"] == [{"googleSearch": {}}]

    async def available_models():
        return [{"id": "gemini-4-argon"}]

    async def unavailable_models():
        return []

    monkeypatch.setattr(antigravity_models, "fetch_available_models", available_models)
    assert asyncio.run(antigravity_models.get_antigravity_models_with_features()) == [
        "gemini-4-argon", "假流式/gemini-4-argon", "流式抗截断/gemini-4-argon"
    ]
    monkeypatch.setattr(antigravity_models, "fetch_available_models", unavailable_models)
    assert asyncio.run(antigravity_models.get_antigravity_models_with_features()) == []
