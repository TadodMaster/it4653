"""Configuration loading helpers.

Loads YAML configs and returns simple dicts/dataclasses.

Placeholder for implementation.
"""

from __future__ import annotations

import yaml


def load_config(path: str) -> dict:
    """Load YAML config file and return a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
