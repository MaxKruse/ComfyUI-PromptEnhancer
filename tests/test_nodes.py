"""Tests for node class structure - verifies preset dropdown and node registration."""

import pytest


def test_prompt_enhancer_has_preset_input():
    """PromptEnhancer node should have a preset dropdown with all presets."""
    from nodes import PromptEnhancer

    input_types = PromptEnhancer.INPUT_TYPES()
    required = input_types.get("required", {})
    assert "preset" in required, "preset should be a required input"

    preset_def = required["preset"]
    assert isinstance(preset_def[0], list), "preset should be a dropdown (list)"
    options = preset_def[0]

    # Should have presets for all target models
    assert len(options) >= 5

    # Display names should include target model labels
    display_str = " ".join(options)
    assert "LTX" in display_str, f"Should have LTX presets in: {options}"
    assert "KREA" in display_str, f"Should have KREA presets in: {options}"


def test_prompt_enhancer_no_target_model_input():
    """PromptEnhancer should NOT have a separate target_model input.

    The target model is determined by the selected preset.
    """
    from nodes import PromptEnhancer

    input_types = PromptEnhancer.INPUT_TYPES()
    required = input_types.get("required", {})
    assert "target_model" not in required, (
        "target_model should be removed - it is derived from the preset"
    )


def test_prompt_enhancer_category_is_generic():
    """Node category should not be LTX-specific."""
    from nodes import PromptEnhancer

    assert PromptEnhancer.CATEGORY is not None
    assert "LTX" not in PromptEnhancer.CATEGORY


def test_prompt_enhancer_display_name_is_generic():
    """Node display name should not be LTX-specific."""
    from nodes import NODE_DISPLAY_NAME_MAPPINGS

    enhancer_key = None
    for key, display in NODE_DISPLAY_NAME_MAPPINGS.items():
        if "Enhancer" in display and "Batch" not in display:
            enhancer_key = key
            break

    assert enhancer_key is not None
    display = NODE_DISPLAY_NAME_MAPPINGS[enhancer_key]
    assert "LTX" not in display


def test_prompt_enhancer_batch_has_preset_input():
    """PromptEnhancerBatch should have a preset dropdown."""
    from nodes import PromptEnhancerBatch

    input_types = PromptEnhancerBatch.INPUT_TYPES()
    required = input_types.get("required", {})
    assert "preset" in required


def test_prompt_enhancer_batch_no_target_model_input():
    """PromptEnhancerBatch should NOT have a separate target_model input."""
    from nodes import PromptEnhancerBatch

    input_types = PromptEnhancerBatch.INPUT_TYPES()
    required = input_types.get("required", {})
    assert "target_model" not in required


def test_node_class_mappings_use_generic_names():
    """Node class mappings should use generic names (not LTX-specific)."""
    from nodes import NODE_CLASS_MAPPINGS

    # Old LTX-specific names should not be present
    assert "LTXPromptEnhancer" not in NODE_CLASS_MAPPINGS
    assert "LTXPromptEnhancerBatch" not in NODE_CLASS_MAPPINGS

    # Generic names should be present
    assert "PromptEnhancer" in NODE_CLASS_MAPPINGS
    assert "PromptEnhancerBatch" in NODE_CLASS_MAPPINGS


def test_preset_dropdown_has_clear_display_names():
    """Preset dropdown should show human-readable names, not raw filenames."""
    from nodes import PromptEnhancer

    input_types = PromptEnhancer.INPUT_TYPES()
    required = input_types.get("required", {})
    options = required["preset"][0]

    # Display names should NOT look like raw filenames
    for opt in options:
        assert not opt.startswith("ltx-t2v-"), f"Display name should not be raw key: {opt}"
        assert not opt.startswith("ltx-i2v-"), f"Display name should not be raw key: {opt}"
        assert not opt == "krea2-t2i", f"Display name should not be raw key: {opt}"

    # Should use "Model - Style" format
    for opt in options:
        assert " - " in opt, f"Display name should use ' - ' separator: {opt}"


def test_preset_options_sorted_alphabetically():
    """Preset options should be sorted for consistent ordering."""
    from nodes import PromptEnhancer

    input_types = PromptEnhancer.INPUT_TYPES()
    required = input_types.get("required", {})
    options = required["preset"][0]

    assert options == sorted(options), f"Preset options should be sorted: {options}"
