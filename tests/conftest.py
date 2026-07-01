"""Pytest configuration for ComfyUI-PromptEnhancer tests."""

import sys
from pathlib import Path

# Add ComfyUI root to sys.path so comfy.* imports work
comfy_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(comfy_root))

# Add the package root to sys.path so imports work
pkg_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pkg_root))


def pytest_ignore_collect(collection_path, config):
    """Ignore __init__.py files that would cause import errors."""
    if collection_path.name == "__init__.py":
        return True
    return False
