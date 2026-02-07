"""OpenAI-based intent classification."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from agentic.config.settings import Settings
from agentic.exceptions import ParseError
from agentic.models.intent import Entity, IntentType, ParsedIntent
from agentic.parser.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from agentic.parser.schemas import INTENT_JSON_SCHEMA


class IntentParser:
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)

    async def parse(self, query: str, context: str = "") -> ParsedIntent:
        if not query.strip():
            raise ParseError("Empty query")

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context or "No previous context.",
            query=query,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": INTENT_JSON_SCHEMA,
                },
                temperature=0.0,
            )
        except Exception as exc:
            raise ParseError(f"OpenAI API error: {exc}") from exc

        raw = response.choices[0].message.content
        if raw is None:
            raise ParseError("OpenAI returned empty content")

        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Malformed JSON from OpenAI: {exc}") from exc

        intent_type = IntentType(data["intent_type"])
        confidence = float(data["confidence"])

        if confidence < 0.5:
            intent_type = IntentType.UNKNOWN

        entities = [
            Entity(name=e["name"], value=e["value"], source=e.get("source", ""))
            for e in data.get("entities", [])
        ]

        return ParsedIntent(
            raw_query=query,
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            reasoning=data.get("reasoning", ""),
        )
