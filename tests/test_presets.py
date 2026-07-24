"""Tests for the preset system - verifies presets exist, load, and have correct metadata."""

import pytest
from presets import list_presets, load_preset, get_default_preset, get_preset_by_key, TARGET_MODEL_LABELS


def test_krea2_t2i_preset_exists():
    """KREA-2 text-to-image preset should exist."""
    content = load_preset("krea2-t2i")
    assert content is not None
    assert len(content) > 100
    assert "text-to-image" in content.lower() or "image" in content.lower()


def test_krea2_t2i_preset_has_expansion_content():
    """KREA-2 preset should contain key expansion instructions."""
    content = load_preset("krea2-t2i")
    assert content is not None
    assert "Faithfulness" in content or "faithfulness" in content
    assert "text-to-image" in content.lower()


def test_krea2_t2i_nsfw_preset_exists():
    """KREA-2 NSFW preset should exist."""
    content = load_preset("krea2-t2i-nsfw")
    assert content is not None
    assert len(content) > 100


def test_target_model_labels_complete():
    """Target model labels should cover all supported models."""
    assert "krea2-t2i" in TARGET_MODEL_LABELS


def test_list_presets_returns_named_tuples():
    """list_presets should return PresetInfo named tuples."""
    presets = list_presets()
    assert len(presets) >= 2

    for p in presets:
        assert hasattr(p, "key")
        assert hasattr(p, "display_name")
        assert hasattr(p, "target_model")
        assert len(p.key) > 0
        assert len(p.display_name) > 0
        assert len(p.target_model) > 0


def test_preset_display_names_included_target_model():
    """Display names should include the target model label."""
    presets = list_presets()
    for p in presets:
        label = TARGET_MODEL_LABELS.get(p.target_model, p.target_model)
        assert label in p.display_name, f"Preset {p.key} display '{p.display_name}' should include '{label}'"


def test_preset_keys_are_unique():
    """Each preset should have a unique key."""
    presets = list_presets()
    keys = [p.key for p in presets]
    assert len(keys) == len(set(keys)), f"Duplicate keys found: {keys}"


def test_preset_display_names_are_unique():
    """Each preset should have a unique display name."""
    presets = list_presets()
    names = [p.display_name for p in presets]
    assert len(names) == len(set(names)), f"Duplicate display names found: {names}"


def test_krea2_presets_have_correct_target():
    """KREA-2 presets should derive target_model=krea2-t2i."""
    presets = list_presets()
    krea2_presets = [p for p in presets if p.key.startswith("krea2-")]
    assert len(krea2_presets) > 0
    for p in krea2_presets:
        assert p.target_model == "krea2-t2i"


def test_get_preset_by_key():
    """get_preset_by_key should return correct metadata."""
    info = get_preset_by_key("krea2-t2i")
    assert info is not None
    assert info.key == "krea2-t2i"
    assert info.target_model == "krea2-t2i"
    assert "KREA" in info.display_name


def test_get_preset_by_key_unknown():
    """get_preset_by_key should return None for unknown keys."""
    assert get_preset_by_key("nonexistent") is None


def test_get_default_preset():
    """get_default_preset should return the first preset alphabetically."""
    default = get_default_preset()
    assert default is not None
    presets = list_presets()
    assert default.key == presets[0].key


def test_load_unknown_preset_returns_none():
    """Loading a non-existent preset should return None."""
    assert load_preset("nonexistent-preset") is None


def test_all_presets_load_without_error():
    """All listed presets should load successfully."""
    all_presets = list_presets()
    assert len(all_presets) >= 2

    for p in all_presets:
        content = load_preset(p.key)
        assert content is not None, f"Preset {p.key} should load"
        assert len(content) > 50, f"Preset {p.key} should have meaningful content"
