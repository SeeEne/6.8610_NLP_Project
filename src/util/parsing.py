"""Shared JSON extraction utilities for LLM responses."""

from __future__ import annotations

import json
import re


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from an LLM response.

    Handles common issues:
      - Markdown code fences (```json ... ```)
      - Leading/trailing prose around JSON
      - Trailing commas inside objects/arrays
      - Curly braces inside JSON string values (string-aware matching)

    Raises ValueError if no valid JSON object can be extracted.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty response")

    # Strip markdown code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Find the outermost { ... } with string-aware brace matching
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:100]}")

    depth = 0
    in_string = False
    escape = False
    end = start
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    # If braces never balanced, the response is likely truncated
    if depth != 0:
        raise ValueError(
            f"Unbalanced braces (depth={depth}) — response likely truncated. "
            f"Last 80 chars: {text[-80:]}"
        )

    text = text[start:end]

    # Try as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: single quotes → double quotes
    return json.loads(text.replace("'", '"'))
