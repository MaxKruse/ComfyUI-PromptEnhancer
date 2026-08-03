"""Tests for node class structure - verifies preset dropdown and node registration."""

import pytest


def _get_input_types(node_class):
    """Get INPUT_TYPES dict from a node class (works for both V1 and V3/io nodes)."""
    return node_class.INPUT_TYPES()


def test_prompt_enhancer_has_preset_input():
    """PromptEnhancer node should have a preset dropdown with all presets."""
    from nodes import PromptEnhancer

    input_types = _get_input_types(PromptEnhancer)
    required = input_types.get("required", {})
    assert "preset" in required, "preset should be a required input"

    preset_def = required["preset"]
    assert isinstance(preset_def[0], str) or isinstance(preset_def[0], list), (
        "preset should be a combo type"
    )

    # For V3/io nodes, the type is "COMBO" and options are in the second element
    if isinstance(preset_def[0], str) and preset_def[0] == "COMBO":
        options = preset_def[1].get("options", [])
    else:
        options = preset_def[0]

    # Should have presets for all target models
    assert len(options) >= 2

    # Display names should include target model labels
    display_str = " ".join(options)
    assert "LTX" in display_str, f"Should have LTX presets in: {options}"
    assert "KREA" in display_str, f"Should have KREA presets in: {options}"


def test_prompt_enhancer_no_target_model_input():
    """PromptEnhancer should NOT have a separate target_model input.

    The target model is determined by the selected preset.
    """
    from nodes import PromptEnhancer

    input_types = _get_input_types(PromptEnhancer)
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})
    assert "target_model" not in required, (
        "target_model should be removed - it is derived from the preset"
    )
    assert "target_model" not in optional


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

    input_types = _get_input_types(PromptEnhancerBatch)
    required = input_types.get("required", {})
    assert "preset" in required


def test_prompt_enhancer_batch_no_target_model_input():
    """PromptEnhancerBatch should NOT have a separate target_model input."""
    from nodes import PromptEnhancerBatch

    input_types = _get_input_types(PromptEnhancerBatch)
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})
    assert "target_model" not in required
    assert "target_model" not in optional


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

    input_types = _get_input_types(PromptEnhancer)
    required = input_types.get("required", {})
    preset_def = required["preset"]

    if isinstance(preset_def[0], str) and preset_def[0] == "COMBO":
        options = preset_def[1].get("options", [])
    else:
        options = preset_def[0]

    # Display names should NOT look like raw filenames
    for opt in options:
        assert not opt.startswith("ltx-t2v-"), f"Display name should not be raw key: {opt}"
        assert not opt.startswith("ltx-i2v-"), f"Display name should not be raw key: {opt}"
        assert not opt == "krea2-t2i", f"Display name should not be raw key: {opt}"

    # Base presets use just the model name (no suffix), custom presets use "Model - Style" format
    for opt in options:
        assert " - " in opt or opt in ["KREA 2 T2I", "LTX 2.3 10Eros I2V", "MiniMax H3"], (
            f"Display name should use ' - ' separator or be a base model name: {opt}"
        )


def test_preset_options_sorted_alphabetically():
    """Preset options should be sorted for consistent ordering."""
    from nodes import PromptEnhancer

    input_types = _get_input_types(PromptEnhancer)
    required = input_types.get("required", {})
    preset_def = required["preset"]

    if isinstance(preset_def[0], str) and preset_def[0] == "COMBO":
        options = preset_def[1].get("options", [])
    else:
        options = preset_def[0]

    assert options == sorted(options), f"Preset options should be sorted: {options}"


def test_prompt_enhancer_has_ref_images_autogrow():
    """PromptEnhancer should have dynamic ref_images input via Autogrow."""
    from nodes import PromptEnhancer

    input_types = _get_input_types(PromptEnhancer)
    optional = input_types.get("optional", {})

    # Autogrow shows as ref_images in INPUT_TYPES (expanded at execution time)
    assert "ref_images" in optional, (
        f"Should have ref_images autogrow input. Optional keys: {list(optional.keys())}"
    )

    # Verify it's an autogrow type
    ref_images_def = optional["ref_images"]
    assert ref_images_def[0] == "COMFY_AUTOGROW_V3", (
        f"ref_images should be autogrow type, got {ref_images_def[0]}"
    )


def test_prompt_enhancer_batch_has_ref_images_autogrow():
    """PromptEnhancerBatch should also have dynamic ref_images input."""
    from nodes import PromptEnhancerBatch

    input_types = _get_input_types(PromptEnhancerBatch)
    optional = input_types.get("optional", {})

    assert "ref_images" in optional, (
        f"Should have ref_images autogrow input. Optional keys: {list(optional.keys())}"
    )

    ref_images_def = optional["ref_images"]
    assert ref_images_def[0] == "COMFY_AUTOGROW_V3", (
        f"ref_images should be autogrow type, got {ref_images_def[0]}"
    )


def test_prompt_enhancer_no_single_reference_image():
    """PromptEnhancer should NOT have the old single reference_image input."""
    from nodes import PromptEnhancer

    input_types = _get_input_types(PromptEnhancer)
    required = input_types.get("required", {})
    optional = input_types.get("optional", {})

    # The old "reference_image" should be gone (replaced by autogrow ref_image_*)
    assert "reference_image" not in required, "Old reference_image should be removed"
    assert "reference_image" not in optional, "Old reference_image should be removed"


def test_prompt_enhancer_returns_string_output():
    """PromptEnhancer should return a STRING output."""
    from nodes import PromptEnhancer

    return_types = PromptEnhancer.RETURN_TYPES
    assert "STRING" in return_types, f"Should return STRING, got {return_types}"


def test_prompt_enhancer_prompt_is_first_input():
    """Prompt should be the first required input for proper bypass pass-through."""
    from nodes import PromptEnhancer

    input_types = _get_input_types(PromptEnhancer)
    required = input_types.get("required", {})
    required_keys = list(required.keys())

    # prompt should be the first required input so it passes through on bypass
    assert required_keys[0] == "prompt", (
        f"prompt should be first required input for bypass, got {required_keys[0]}"
    )
