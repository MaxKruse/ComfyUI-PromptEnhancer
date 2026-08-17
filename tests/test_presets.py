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
    assert "User Prompt Is Ground Truth" in content or "ground truth" in content.lower()
    assert "text-to-image" in content.lower()


def test_krea2_t2i_preset_has_nsfw_directives():
    """KREA-2 preset should contain NSFW directives for auto-detection."""
    content = load_preset("krea2-t2i")
    assert content is not None
    assert "NSFW" in content
    assert "anatomical" in content.lower()


def test_krea2_t2i_preset_has_snofs_directives():
    """KREA-2 preset should contain SNOFS v1.3D LoRA directives."""
    content = load_preset("krea2-t2i")
    assert content is not None
    assert "SNOFS" in content
    # Trained vocabulary from the LoRA author's term list
    for term in ("blowjob", "cunnilingus", "doggystyle position", "penis", "vagina", "cum"):
        assert term in content
    # Natural-language (full sentence) prompting rule
    assert "full sentence" in content.lower()


def test_krea2_t2i_preset_photo_wording_rule():
    """KREA-2 preset must require literal photo wording and forbid 'photorealistic'."""
    content = load_preset("krea2-t2i")
    assert content is not None
    assert "photograph" in content
    # 'photorealistic' may only appear as a forbidden token
    lowered = content.lower()
    assert "photorealistic" in lowered
    assert '"photorealistic"' in lowered


def test_ltx2_3_i2v_preset_exists():
    """LTX 2.3 10Eros image-to-video preset should exist."""
    content = load_preset("ltx2.3-10eros-i2v")
    assert content is not None
    assert len(content) > 100
    assert "image-to-video" in content.lower() or "video" in content.lower()


def test_ltx2_3_i2v_preset_has_nsfw_directives():
    """LTX 2.3 preset should contain NSFW directives for auto-detection."""
    content = load_preset("ltx2.3-10eros-i2v")
    assert content is not None
    assert "NSFW" in content
    assert "anatomical" in content.lower()


def test_no_separate_nsfw_presets():
    """NSFW presets should be merged into main presets, not separate files."""
    assert load_preset("krea2-t2i-nsfw") is None
    assert load_preset("ltx2.3-10eros-i2v-nsfw") is None


def test_minimax_h3_r2v_preset_exists():
    """MiniMax H3 R2V preset should exist."""
    content = load_preset("minimax-h3-r2v")
    assert content is not None
    assert len(content) > 100
    assert "video" in content.lower()


def test_minimax_h3_r2v_preset_has_key_content():
    """MiniMax H3 R2V preset should contain key video prompting instructions."""
    content = load_preset("minimax-h3-r2v")
    assert content is not None
    assert "audio" in content.lower()
    assert "MM:SS" in content
    assert "camera" in content.lower()
    assert "subject_definitions" in content.lower()
    assert "detailed_description" in content.lower()


def test_minimax_h3_r2v_preset_is_r2v_only():
    """MiniMax H3 R2V preset should be Reference-to-Video only."""
    content = load_preset("minimax-h3-r2v")
    assert content is not None
    assert "reference-to-video" in content.lower() or "r2v" in content.lower()
    assert "<Subject" in content
    assert "<Picture" in content
    assert "<Audio" in content


def test_minimax_h3_r2v_preset_has_nsfw_directives():
    """MiniMax H3 R2V preset should contain NSFW directives for auto-detection."""
    content = load_preset("minimax-h3-r2v")
    assert content is not None
    assert "NSFW" in content
    assert "anatomical" in content.lower()


def test_minimax_presets_have_correct_target():
    """MiniMax H3 presets should derive target_model=minimax-h3."""
    presets = list_presets()
    minimax_presets = [p for p in presets if p.key.startswith("minimax-h3")]
    assert len(minimax_presets) > 0
    for p in minimax_presets:
        assert p.target_model == "minimax-h3"


def test_target_model_labels_complete():
    """Target model labels should cover all supported models."""
    assert "krea2-t2i" in TARGET_MODEL_LABELS
    assert "ltx2.3-10eros-i2v" in TARGET_MODEL_LABELS
    assert "minimax-h3" in TARGET_MODEL_LABELS


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


def test_ltx_presets_have_correct_target():
    """LTX presets should derive target_model=ltx2.3-10eros-i2v."""
    presets = list_presets()
    ltx_presets = [p for p in presets if p.key.startswith("ltx2.3-")]
    assert len(ltx_presets) > 0
    for p in ltx_presets:
        assert p.target_model == "ltx2.3-10eros-i2v"


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
