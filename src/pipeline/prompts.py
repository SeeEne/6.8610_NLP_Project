"""Centralized prompt loader — reads from config/prompts.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_PROMPTS: dict | None = None


def _load_prompts() -> dict:
    global _PROMPTS
    if _PROMPTS is None:
        with open(_CONFIG_DIR / "prompts.yaml", "r") as f:
            _PROMPTS = yaml.safe_load(f)
    return _PROMPTS


def get_prompt(path: str) -> str:
    """Get a prompt by dot-separated path.

    Example:
        get_prompt("anchor_selection.ambiguity_classification.system")
        get_prompt("anchor_selection.ambiguity_definitions")
    """
    prompts = _load_prompts()
    keys = path.split(".")
    val = prompts
    for key in keys:
        val = val[key]
    return val


def render_prompt(path: str, **kwargs) -> str:
    """Get a prompt and fill in template variables.

    Example:
        render_prompt("anchor_selection.ambiguity_classification.task",
                      prompt="Write a function...",
                      ambiguity_definitions=get_prompt("anchor_selection.ambiguity_definitions"))
    """
    template = get_prompt(path)
    return template.format(**kwargs)


def load_pipeline_config() -> dict:
    """Load config/pipeline.yaml."""
    with open(_CONFIG_DIR / "pipeline.yaml", "r") as f:
        return yaml.safe_load(f)
