"""Preset file discovery and loading for prompt enhancement system prompts.

Presets are organized by target model prefix:
  - ltx-t2v-*   : LTX 2.3 text-to-video presets
  - ltx-i2v-*   : LTX 2.3 image-to-video presets
  - krea2-t2i-* : KREA-2 text-to-image presets
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


class PresetInfo(NamedTuple):
    """Metadata for a single preset option."""

    key: str  # Internal key used in workflows (backward compat with filename)
    display_name: str  # Human-readable label shown in the dropdown
    target_model: str  # Target model ID derived from the preset prefix


# Mapping of target_model prefixes to their human-readable labels.
TARGET_MODEL_LABELS: dict[str, str] = {
    "ltx-t2v": "LTX 2.3 T2V",
    "ltx-i2v": "LTX 2.3 I2V",
    "krea2-t2i": "KREA 2 T2I",
}

# Mapping from preset key -> display suffix (the part after the target prefix).
# Add new entries here when adding preset files.
_PRESET_DISPLAY_NAMES: dict[str, str] = {
    "ltx-t2v-cinematic": "Cinematic",
    "ltx-t2v-nsfw": "NSFW",
    "ltx-i2v-motion": "Motion",
    "ltx-i2v-nsfw": "NSFW",
    "krea2-t2i": "General",
    "krea2-t2i-nsfw": "NSFW",
}


def _derive_target_model(preset_key: str) -> str:
    """Extract the target model ID from a preset key.

    Matches the longest known prefix first (e.g. 'ltx-t2v' from 'ltx-t2v-cinematic').
    Falls back to the key itself if no prefix matches (e.g. 'krea2-t2i' == 'krea2-t2i').
    """
    # Sort prefixes longest-first so 'ltx-t2v' matches before 'ltx'
    for prefix in sorted(TARGET_MODEL_LABELS.keys(), key=len, reverse=True):
        if preset_key == prefix or preset_key.startswith(prefix + "-"):
            return prefix
    return preset_key


def list_presets() -> list[PresetInfo]:
    """Return sorted list of all available presets with display metadata.

    Presets discovered from the filesystem are enriched with display names
    and target model info. Presets without an explicit display name use
    the raw filename suffix.
    """
    if not PRESETS_DIR.is_dir():
        return []

    results: list[PresetInfo] = []
    for path in sorted(PRESETS_DIR.glob("*.txt")):
        key = path.stem
        target = _derive_target_model(key)
        label = TARGET_MODEL_LABELS.get(target, target)

        if key in _PRESET_DISPLAY_NAMES:
            display = f"{label} - {_PRESET_DISPLAY_NAMES[key]}"
        else:
            # Derive suffix: strip the target prefix (and trailing dash)
            suffix = key
            if key == target:
                suffix = "General"
            elif key.startswith(target + "-"):
                suffix = key[len(target) + 1 :]
            display = f"{label} - {suffix}"

        results.append(PresetInfo(key=key, display_name=display, target_model=target))

    # Sort by display name for consistent alphabetical ordering in the UI
    results.sort(key=lambda p: p.display_name)
    return results


def load_preset(name: str) -> str | None:
    """Load a preset file by key (filename without .txt extension). Returns None if not found."""
    preset_path = PRESETS_DIR / f"{name}.txt"
    if not preset_path.is_file():
        return None
    return preset_path.read_text(encoding="utf-8").strip()


def get_preset_by_key(key: str) -> PresetInfo | None:
    """Look up preset metadata by key."""
    for p in list_presets():
        if p.key == key:
            return p
    return None


def get_default_preset() -> PresetInfo | None:
    """Return the first available preset (sorted alphabetically by key)."""
    presets = list_presets()
    return presets[0] if presets else None
