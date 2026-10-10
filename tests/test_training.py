"""Dataset validation and leakage prevention; these tests don't measure food accuracy."""

from collections import Counter
import json

import pytest

from hotpot.training import CLASS_IDS, read_samples, split_by_session
from hotpot.ingredients import SAMPLE_LABELS


def test_training_uses_the_same_class_order_as_the_app():
    assert CLASS_IDS == SAMPLE_LABELS
    assert {"udon_noodles", "shiitake_mushrooms", "beef_meatballs"}.issubset(CLASS_IDS)


def session_samples():
    return [
        {"label": label, "session_id": session, "path": f"{label}-{session}-{index}.jpg"}
        for label in CLASS_IDS for session in ("meal-a", "meal-b", "meal-c") for index in range(4)
    ]


def test_split_keeps_entire_sessions_together_and_all_classes_present():
    training, validation = split_by_session(session_samples())
    assert {sample["session_id"] for sample in training}.isdisjoint(
        {sample["session_id"] for sample in validation}
    )
    assert set(Counter(s["label"] for s in training)) == set(CLASS_IDS)
    assert set(Counter(s["label"] for s in validation)) == set(CLASS_IDS)
    assert split_by_session(session_samples()) == (training, validation)


def test_same_session_cannot_be_split_into_adjacent_frames():
    samples = [{**s, "session_id": "one-recording"} for s in session_samples()]
    with pytest.raises(ValueError, match="at least 2 separate collection sessions"):
        split_by_session(samples)


def test_training_requires_background_examples():
    samples = [s for s in session_samples() if s["label"] != "background"]
    with pytest.raises(ValueError, match="background: 0 images"):
        split_by_session(samples)


@pytest.mark.parametrize("missing_label", ["udon_noodles", "shiitake_mushrooms", "beef_meatballs"])
def test_training_requires_examples_for_new_ingredients(missing_label):
    samples = [s for s in session_samples() if s["label"] != missing_label]
    with pytest.raises(ValueError, match=f"{missing_label}: 0 images"):
        split_by_session(samples)


def test_manifest_rejects_paths_outside_data_directory(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (tmp_path / "outside.jpg").write_bytes(b"image")
    (data_dir / "samples.jsonl").write_text(json.dumps({
        "label": "beef_rolls", "session_id": "a", "path": "../outside.jpg"
    }) + "\n")
    with pytest.raises(ValueError, match="outside data directory"):
        read_samples(data_dir)


def test_manifest_reports_broken_line_and_duplicate_images(tmp_path):
    (tmp_path / "sample.jpg").write_bytes(b"image")
    record = {"label": "soft_tofu", "session_id": "a", "path": "sample.jpg"}
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps(record) + "\n{bad json}\n")
    with pytest.raises(ValueError, match="manifest line 2"):
        read_samples(tmp_path)
    manifest.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate image path"):
        read_samples(tmp_path)


def test_evaluator_refuses_training_and_calibration_sessions(monkeypatch, tmp_path):
    from scripts import evaluate as evaluator

    class Model:
        def __init__(self, *_):
            pass
        def status(self):
            return {"ready": True, "mode": "finetuned", "metadata": {
                "train_sessions": ["meal-a"], "validation_sessions": ["meal-b"]
            }}

    monkeypatch.setattr(evaluator, "Recognizer", Model)
    monkeypatch.setattr(evaluator, "read_samples", lambda _: session_samples())
    for session in ("meal-a", "meal-b"):
        with pytest.raises(ValueError, match="already used for training or calibration"):
            evaluator.evaluate(tmp_path, tmp_path, [session])
