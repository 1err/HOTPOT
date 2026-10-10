"""Vocabulary compatibility checks; prompt coverage is not evidence of food accuracy."""

import pytest

from hotpot.ingredients import INGREDIENT_IDS
from hotpot.recognition import (
    CLIP_CHECKPOINT, PROMPTS, Recognizer, validate_ingredient_checkpoint,
)
from hotpot.training import CHECKPOINT_NAME, CLASS_IDS
from scripts.import_images import CHINESE_NAMES


LEGACY_LABELS = (
    "beef_rolls", "crab_sticks", "napa_cabbage", "lotus_root", "soft_tofu", "wide_noodles", "background"
)


def checkpoint(labels):
    return {"format_version": 1, "arch": "mobilenet_v3_small", "class_ids": list(labels)}


def test_prompt_groups_cover_the_canonical_ingredient_vocabulary():
    assert set(PROMPTS) == set(INGREDIENT_IDS)
    assert all(len(PROMPTS[label]) >= 3 for label in INGREDIENT_IDS)


def test_chinese_import_folder_names_include_new_ingredients():
    assert CHINESE_NAMES["乌冬面"] == "udon_noodles"
    assert CHINESE_NAMES["香菇"] == "shiitake_mushrooms"
    assert CHINESE_NAMES["牛肉丸"] == "beef_meatballs"


def test_compatible_checkpoint_retains_canonical_label_order():
    validate_ingredient_checkpoint(checkpoint(CLASS_IDS))
    with pytest.raises(ValueError, match="label order differs"):
        validate_ingredient_checkpoint(checkpoint(tuple(reversed(CLASS_IDS))))


def test_six_ingredient_checkpoint_requires_retraining():
    with pytest.raises(ValueError, match="udon_noodles, shiitake_mushrooms, beef_meatballs"):
        validate_ingredient_checkpoint(checkpoint(LEGACY_LABELS))


def test_obsolete_checkpoint_falls_back_with_a_visible_explanation(monkeypatch, tmp_path):
    (tmp_path / CHECKPOINT_NAME).touch()
    (tmp_path / CLIP_CHECKPOINT).touch()

    def load_old(self, _):
        validate_ingredient_checkpoint(checkpoint(LEGACY_LABELS))

    def load_baseline(self, _):
        self._mode = "clip"
        self._model = object()
        self._message = "Zero-shot baseline"
        self._metadata = {"ingredient_training": False}

    monkeypatch.setattr(Recognizer, "_load_finetuned", load_old)
    monkeypatch.setattr(Recognizer, "_load_clip", load_baseline)
    status = Recognizer(tmp_path / "data", tmp_path).status()
    assert status["state"] == "ready" and status["mode"] == "clip"
    assert status["metadata"]["ingredient_training"] is False
    assert "Retrain" in status["message"]
    assert "udon_noodles" in status["metadata"]["fallback_reason"]
    assert status["warnings"]
