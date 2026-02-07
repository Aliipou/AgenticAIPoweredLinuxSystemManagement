"""Prompt templates for OpenAI intent classification."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a Linux system management assistant. Your job is to classify user requests
into one of these intent types:

- FOCUS: The user wants to reduce distractions (kill/suspend browsers, media players, chat apps, etc.)
- UPDATE: The user wants to update or install system packages.
- CLEAN_MEMORY: The user wants to free up RAM, clear caches, or kill memory-hogging processes.
- UNKNOWN: The request does not match any known intent.

Respond ONLY with valid JSON matching the provided schema. Extract any relevant entities
(process names, package names, service names) from the query.

Provide a confidence score between 0.0 and 1.0 indicating how sure you are of the classification.
If confidence is below 0.5, use UNKNOWN.
"""

USER_PROMPT_TEMPLATE = """\
{context}

User request: {query}

Classify this request and extract entities.
"""
