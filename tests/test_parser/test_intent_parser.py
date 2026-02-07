"""Brutal tests for intent_parser, schemas, and prompt_templates."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic.exceptions import ParseError
from agentic.models.intent import IntentType
from agentic.parser.intent_parser import IntentParser
from agentic.parser.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from agentic.parser.schemas import INTENT_JSON_SCHEMA


class TestPromptTemplates:
    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 50
        assert "FOCUS" in SYSTEM_PROMPT
        assert "UPDATE" in SYSTEM_PROMPT
        assert "CLEAN_MEMORY" in SYSTEM_PROMPT
        assert "UNKNOWN" in SYSTEM_PROMPT

    def test_user_prompt_template_has_placeholders(self):
        assert "{context}" in USER_PROMPT_TEMPLATE
        assert "{query}" in USER_PROMPT_TEMPLATE

    def test_user_prompt_template_renders(self):
        rendered = USER_PROMPT_TEMPLATE.format(context="some context", query="do stuff")
        assert "some context" in rendered
        assert "do stuff" in rendered


class TestSchemas:
    def test_schema_has_required_structure(self):
        assert INTENT_JSON_SCHEMA["name"] == "parsed_intent"
        assert INTENT_JSON_SCHEMA["strict"] is True
        schema = INTENT_JSON_SCHEMA["schema"]
        assert "intent_type" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert "entities" in schema["properties"]
        assert "reasoning" in schema["properties"]

    def test_schema_intent_type_enum(self):
        enum_vals = INTENT_JSON_SCHEMA["schema"]["properties"]["intent_type"]["enum"]
        assert set(enum_vals) == {"FOCUS", "UPDATE", "CLEAN_MEMORY", "UNKNOWN"}

    def test_schema_entities_structure(self):
        items = INTENT_JSON_SCHEMA["schema"]["properties"]["entities"]["items"]
        assert "name" in items["properties"]
        assert "value" in items["properties"]
        assert "source" in items["properties"]

    def test_schema_required_fields(self):
        required = INTENT_JSON_SCHEMA["schema"]["required"]
        assert "intent_type" in required
        assert "confidence" in required
        assert "entities" in required
        assert "reasoning" in required


class TestIntentParser:
    @pytest.fixture
    def parser(self, mock_settings):
        client = AsyncMock()
        return IntentParser(mock_settings, client=client)

    @pytest.mark.asyncio
    async def test_parse_focus_intent(self, parser, mock_openai_response):
        data = {
            "intent_type": "FOCUS",
            "confidence": 0.95,
            "entities": [{"name": "process", "value": "chrome", "source": "chrome"}],
            "reasoning": "User wants to focus",
        }
        parser._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(data)
        )

        result = await parser.parse("close chrome so I can focus")
        assert result.intent_type == IntentType.FOCUS
        assert result.confidence == 0.95
        assert len(result.entities) == 1
        assert result.entities[0].value == "chrome"

    @pytest.mark.asyncio
    async def test_parse_update_intent(self, parser, mock_openai_response):
        data = {
            "intent_type": "UPDATE",
            "confidence": 0.9,
            "entities": [],
            "reasoning": "System update requested",
        }
        parser._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(data)
        )

        result = await parser.parse("update my packages")
        assert result.intent_type == IntentType.UPDATE

    @pytest.mark.asyncio
    async def test_parse_clean_memory_intent(self, parser, mock_openai_response):
        data = {
            "intent_type": "CLEAN_MEMORY",
            "confidence": 0.85,
            "entities": [{"name": "process", "value": "electron", "source": "electron"}],
            "reasoning": "Free RAM",
        }
        parser._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(data)
        )

        result = await parser.parse("free up memory kill electron")
        assert result.intent_type == IntentType.CLEAN_MEMORY
        assert result.entities[0].value == "electron"

    @pytest.mark.asyncio
    async def test_parse_unknown_intent(self, parser, mock_openai_response):
        data = {
            "intent_type": "UNKNOWN",
            "confidence": 0.3,
            "entities": [],
            "reasoning": "Not a system request",
        }
        parser._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(data)
        )

        result = await parser.parse("tell me a joke")
        assert result.intent_type == IntentType.UNKNOWN

    @pytest.mark.asyncio
    async def test_low_confidence_falls_back_to_unknown(self, parser, mock_openai_response):
        data = {
            "intent_type": "FOCUS",
            "confidence": 0.3,
            "entities": [],
            "reasoning": "Low confidence",
        }
        parser._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(data)
        )

        result = await parser.parse("maybe focus?")
        assert result.intent_type == IntentType.UNKNOWN
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_empty_input_raises_parse_error(self, parser):
        with pytest.raises(ParseError, match="Empty query"):
            await parser.parse("")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises_parse_error(self, parser):
        with pytest.raises(ParseError, match="Empty query"):
            await parser.parse("   ")

    @pytest.mark.asyncio
    async def test_api_error_raises_parse_error(self, parser):
        parser._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API timeout")
        )
        with pytest.raises(ParseError, match="OpenAI API error"):
            await parser.parse("focus mode")

    @pytest.mark.asyncio
    async def test_malformed_json_raises_parse_error(self, parser):
        message = MagicMock()
        message.content = "not valid json {{"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]

        parser._client.chat.completions.create = AsyncMock(return_value=response)

        with pytest.raises(ParseError, match="Malformed JSON"):
            await parser.parse("focus")

    @pytest.mark.asyncio
    async def test_none_content_raises_parse_error(self, parser):
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]

        parser._client.chat.completions.create = AsyncMock(return_value=response)

        with pytest.raises(ParseError, match="empty content"):
            await parser.parse("focus")

    @pytest.mark.asyncio
    async def test_parse_with_context(self, parser, mock_openai_response):
        data = {
            "intent_type": "FOCUS",
            "confidence": 0.92,
            "entities": [],
            "reasoning": "Focus with context",
        }
        parser._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(data)
        )

        result = await parser.parse("focus", context="Previous: user asked to update")
        assert result.intent_type == IntentType.FOCUS

    @pytest.mark.asyncio
    async def test_entities_without_source_default(self, parser, mock_openai_response):
        data = {
            "intent_type": "FOCUS",
            "confidence": 0.9,
            "entities": [{"name": "process", "value": "slack"}],
            "reasoning": "Focus",
        }
        parser._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(data)
        )

        result = await parser.parse("close slack")
        assert result.entities[0].source == ""

    @pytest.mark.asyncio
    async def test_parse_sets_raw_query(self, parser, mock_openai_response):
        data = {
            "intent_type": "UPDATE",
            "confidence": 0.8,
            "entities": [],
            "reasoning": "Update",
        }
        parser._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(data)
        )

        result = await parser.parse("update everything please")
        assert result.raw_query == "update everything please"

    @pytest.mark.asyncio
    async def test_parse_reasoning_stored(self, parser, mock_openai_response):
        data = {
            "intent_type": "FOCUS",
            "confidence": 0.9,
            "entities": [],
            "reasoning": "User explicitly asked to focus",
        }
        parser._client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(data)
        )

        result = await parser.parse("focus mode")
        assert result.reasoning == "User explicitly asked to focus"
