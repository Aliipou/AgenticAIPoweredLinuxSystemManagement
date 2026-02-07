"""JSON schema for OpenAI structured output."""

from __future__ import annotations

INTENT_JSON_SCHEMA: dict = {
    "name": "parsed_intent",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "intent_type": {
                "type": "string",
                "enum": ["FOCUS", "UPDATE", "CLEAN_MEMORY", "UNKNOWN"],
                "description": "The classified intent type.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0.",
            },
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Entity label (e.g., 'process', 'package', 'service').",
                        },
                        "value": {
                            "type": "string",
                            "description": "The extracted value.",
                        },
                        "source": {
                            "type": "string",
                            "description": "The substring from the query.",
                        },
                    },
                    "required": ["name", "value", "source"],
                    "additionalProperties": False,
                },
                "description": "Extracted entities from the query.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this intent was chosen.",
            },
        },
        "required": ["intent_type", "confidence", "entities", "reasoning"],
        "additionalProperties": False,
    },
}
