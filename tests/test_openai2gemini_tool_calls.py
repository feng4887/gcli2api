"""Regression coverage for selectively ported upstream tool-call fixes."""

import asyncio
import json

import pytest

from src.converter.openai2gemini import (
    _STREAM_TOOL_INDEX,
    _STREAM_TOOL_INDEX_MAX_ENTRIES,
    convert_gemini_to_openai_stream,
    convert_openai_to_gemini_request,
)


@pytest.fixture(autouse=True)
def clear_stream_state(monkeypatch):
    monkeypatch.setenv("COMPATIBILITY_MODE", "false")
    _STREAM_TOOL_INDEX.clear()
    yield
    _STREAM_TOOL_INDEX.clear()


def _choice(parts, response_id="test-stream", finish_reason=None):
    candidate = {"content": {"role": "model", "parts": parts}}
    if finish_reason:
        candidate["finishReason"] = finish_reason
    result = convert_gemini_to_openai_stream(
        "data: " + json.dumps({"candidates": [candidate]}), "gemini-4-argon", response_id
    )
    return json.loads(result.removeprefix("data: "))["choices"][0]


def _call(city):
    return {"functionCall": {"name": "weather", "args": {"city": city}}}


def test_parallel_tool_indices_survive_client_assembly():
    assembled = {}
    for city in ("Beijing", "Shanghai", "Guangzhou"):
        choice = _choice([_call(city)])
        assert choice["finish_reason"] is None
        call = choice["delta"]["tool_calls"][0]
        assembled.setdefault(call["index"], "")
        assembled[call["index"]] += call["function"]["arguments"]
    assert list(assembled) == [0, 1, 2]
    assert [json.loads(args)["city"] for args in assembled.values()] == [
        "Beijing", "Shanghai", "Guangzhou"
    ]
    assert _choice([], finish_reason="STOP")["finish_reason"] == "tool_calls"
    assert not _STREAM_TOOL_INDEX


def test_one_chunk_and_separate_responses_have_independent_indices():
    first = _choice([{"text": "Checking"}, _call("A"), _call("B")], "a")
    second = _choice([_call("C")], "b")
    third = _choice([_call("D")], "a")
    assert [call["index"] for call in first["delta"]["tool_calls"]] == [0, 1]
    assert second["delta"]["tool_calls"][0]["index"] == 0
    assert third["delta"]["tool_calls"][0]["index"] == 2


@pytest.mark.parametrize("reason,expected", [("MAX_TOKENS", "length"), ("SAFETY", "content_filter")])
def test_non_stop_finish_reasons_are_preserved(reason, expected):
    _choice([_call("A")])
    assert _choice([], finish_reason=reason)["finish_reason"] == expected
    assert not _STREAM_TOOL_INDEX


def test_text_stream_does_not_finish_early():
    assert _choice([{"text": "Hello"}])["finish_reason"] is None
    assert _choice([], finish_reason="STOP")["finish_reason"] == "stop"


def test_abandoned_stream_state_is_bounded():
    for index in range(_STREAM_TOOL_INDEX_MAX_ENTRIES + 1):
        _choice([_call("A")], f"abandoned-{index}")
    assert len(_STREAM_TOOL_INDEX) == _STREAM_TOOL_INDEX_MAX_ENTRIES
    assert "abandoned-0" not in _STREAM_TOOL_INDEX


@pytest.mark.parametrize("arguments,expected", [
    ('{"city":"A"}{"city":"B"}', {"city": "A"}), ("invalid", {}),
])
def test_malformed_history_keeps_call_and_merges_adjacent_user_turns(arguments, expected):
    result = asyncio.run(convert_openai_to_gemini_request({
        "model": "gemini-4-argon",
        "messages": [
            {"role": "user", "content": "Check weather"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-a", "type": "function",
                "function": {"name": "weather", "arguments": arguments},
            }]},
            {"role": "tool", "tool_call_id": "call-a", "name": "weather", "content": "Failed"},
            {"role": "user", "content": "Try again"},
        ],
    }))
    assert [content["role"] for content in result["contents"]] == ["user", "model", "user"]
    call = result["contents"][1]["parts"][0]["functionCall"]
    response = result["contents"][2]["parts"][0]["functionResponse"]
    assert call["name"] == response["name"] == "weather"
    assert call["id"] == response["id"] == "call-a"
    assert call["args"] == expected
    assert result["contents"][2]["parts"][-1] == {"text": "Try again"}
