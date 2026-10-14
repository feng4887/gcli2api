"""Verify Claude 5.5 discovery and requests without using live credentials."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from src.api import antigravity as api
from src.converter.anthropic2gemini import anthropic_to_gemini_request
from src.converter.gemini_fix import normalize_gemini_request
from src.converter.openai2gemini import convert_openai_to_gemini_request
from src.router.antigravity import model_list
from src.utils import get_base_model_from_feature_model


CLAUDE_MODELS = [
    "claude-sonnet-5-5", "claude-sonnet-5-5-thinking",
    "claude-opus-5-5", "claude-opus-5-5-thinking",
]


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch):
    monkeypatch.setenv("COMPATIBILITY_MODE", "false")
    monkeypatch.setenv("RETURN_THOUGHTS_TO_FRONTEND", "true")


@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_claude_5_5_keeps_exact_model_and_removes_prefill(model):
    request = {
        "model": model,
        "generationConfig": {
            "thinkingConfig": {"thinkingBudget": 1024, "thinkingLevel": "HIGH"},
            "temperature": 0.7, "topP": 0.9, "topK": 40,
            "presencePenalty": 0.5, "frequencyPenalty": 0.5, "stopSequences": ["stop"],
            "maxOutputTokens": 4096,
        },
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Prefill"}]},
        ],
    }
    original = copy.deepcopy(request)
    result = asyncio.run(normalize_gemini_request(request, mode="antigravity"))
    assert result["model"] == model
    assert result["contents"] == [original["contents"][0]]
    assert result["generationConfig"] == {
        "thinkingConfig": {"includeThoughts": True}, "maxOutputTokens": 4096,
    }
    assert request == original


@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_claude_5_5_preserves_signed_tool_history(model):
    contents = [
        {"role": "user", "parts": [{"text": "Check"}]},
        {"role": "model", "parts": [
            {"text": "reasoning", "thought": True, "thoughtSignature": "signed-history"},
            {"functionCall": {"name": "lookup", "args": {}, "id": "call-1"}},
        ]},
        {"role": "user", "parts": [{"functionResponse": {
            "name": "lookup", "id": "call-1", "response": {"value": 42},
        }}]},
    ]
    result = asyncio.run(normalize_gemini_request({
        "model": model, "contents": copy.deepcopy(contents),
        "tools": [{"functionDeclarations": [{
            "name": "lookup", "parametersJsonSchema": {"type": "object", "properties": {}},
        }]}],
    }, mode="antigravity"))
    assert result["model"] == model
    assert result["contents"] == contents
    assert result["generationConfig"]["thinkingConfig"] == {"includeThoughts": True}
    declaration = result["tools"][0]["functionDeclarations"][0]
    assert declaration["parameters"] == {"type": "object", "properties": {}}
    assert "parametersJsonSchema" not in declaration


@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_claude_5_5_does_not_inject_synthetic_thoughts(model):
    contents = [
        {"role": "user", "parts": [{"text": "First question"}]},
        {"role": "model", "parts": [{"text": "First answer"}]},
        {"role": "user", "parts": [{"text": "Next question"}]},
    ]
    result = asyncio.run(normalize_gemini_request({
        "model": model, "contents": copy.deepcopy(contents),
    }, mode="antigravity"))
    assert result["model"] == model
    assert result["contents"] == contents


@pytest.mark.parametrize("model", CLAUDE_MODELS)
@pytest.mark.parametrize("prefix", ["", "假流式/", "流式抗截断/"])
def test_claude_5_5_openai_and_anthropic_conversion(model, prefix):
    real_model = get_base_model_from_feature_model(prefix + model)
    payload = {"model": real_model, "messages": [{"role": "user", "content": "Hello"}]}
    openai = asyncio.run(convert_openai_to_gemini_request({**payload, "max_tokens": 2048}))
    anthropic = asyncio.run(anthropic_to_gemini_request({
        **payload, "max_tokens": 2048, "thinking": {"type": "adaptive"},
    }))
    for converted in (openai, anthropic):
        converted["model"] = real_model
        result = asyncio.run(normalize_gemini_request(converted, mode="antigravity"))
        assert result["model"] == model
        assert result["generationConfig"]["maxOutputTokens"] == 2048
        assert result["generationConfig"]["thinkingConfig"] == {"includeThoughts": True}
        assert not {"temperature", "topP", "topK"} & result["generationConfig"].keys()


def test_claude_5_5_dynamic_discovery_and_both_model_formats(monkeypatch):
    backend = {model: {} for model in CLAUDE_MODELS}
    monkeypatch.setattr(api.credential_manager, "get_valid_credential", AsyncMock(
        return_value=("test-credential", {"access_token": "test-token"}),
    ))
    monkeypatch.setattr(api, "get_antigravity_api_url", AsyncMock(return_value="https://test.invalid"))
    monkeypatch.setattr(api, "post_async", AsyncMock(
        return_value=httpx.Response(200, json={"models": backend}),
    ))
    available = asyncio.run(api.fetch_available_models())
    assert {entry["id"] for entry in available} == set(CLAUDE_MODELS)
    monkeypatch.setattr(model_list, "fetch_available_models", AsyncMock(return_value=available))
    for endpoint, key, field in (
        (model_list.list_openai_models, "data", "id"),
        (model_list.list_gemini_models, "models", "name"),
    ):
        body = json.loads(asyncio.run(endpoint("test")).body)
        ids = {entry[field].removeprefix("models/") for entry in body[key]}
        for model in CLAUDE_MODELS:
            for prefix in ("", "假流式/", "流式抗截断/"):
                assert prefix + model in ids
    monkeypatch.setattr(model_list, "fetch_available_models", AsyncMock(return_value=[]))
    assert asyncio.run(model_list.get_antigravity_models_with_features()) == []


@pytest.mark.parametrize("requested,expected", [
    ("claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("claude-sonnet-4-6-thinking", "claude-sonnet-4-6"),
    ("claude-opus-4-6", "claude-opus-4-6-thinking"),
    ("claude-haiku-4-5", "gemini-2.5-flash"),
])
def test_existing_claude_aliases_are_unchanged(requested, expected):
    result = asyncio.run(normalize_gemini_request({
        "model": requested, "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
    }, mode="antigravity"))
    assert result["model"] == expected
