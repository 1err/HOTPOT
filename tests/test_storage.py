import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from hotpot.ingredients import DEFAULT_DURATIONS, SAMPLE_LABELS
from hotpot.storage import Store, decode_image


def test_duration_updates_persist_and_do_not_discard_other_values(tmp_path):
    store = Store(tmp_path)
    assert store.durations() == DEFAULT_DURATIONS
    store.set_durations({"beef_rolls": 12, "crab_sticks": 90})
    store.set_durations({"beef_rolls": None, "wide_noodles": 180})
    persisted = Store(tmp_path).durations()
    assert persisted["beef_rolls"] is None
    assert persisted["crab_sticks"] == 90
    assert persisted["wide_noodles"] == 180


def test_existing_six_ingredient_config_preserves_overrides_and_adds_new_defaults(tmp_path):
    store = Store(tmp_path)
    original = {
        "beef_rolls": 25, "crab_sticks": None, "napa_cabbage": 151,
        "lotus_root": 301, "soft_tofu": 301, "wide_noodles": 481,
    }
    store.config_path.write_text(json.dumps({"durations": original}), encoding="utf-8")
    new_defaults = {"udon_noodles": 180, "shiitake_mushrooms": 300, "beef_meatballs": 480}
    assert store.durations() == {**original, **new_defaults}
    store.set_durations({"udon_noodles": 200})
    expected = {**original, **new_defaults, "udon_noodles": 200}
    assert Store(tmp_path).durations() == expected
    assert json.loads(store.config_path.read_text(encoding="utf-8"))["durations"] == expected


@pytest.mark.parametrize("updates", [
    {"unknown": 10}, {"beef_rolls": 0}, {"beef_rolls": -1},
    {"beef_rolls": 7201}, {"beef_rolls": 2.5},
    {"beef_rolls": True}, {"beef_rolls": "10"},
])
def test_invalid_duration_update_cannot_partially_change_config(tmp_path, updates):
    store = Store(tmp_path)
    store.set_durations({"soft_tofu": 40})
    with pytest.raises(ValueError):
        store.set_durations({"soft_tofu": 80, **updates})
    assert store.durations()["soft_tofu"] == 40


def test_capture_concurrency_and_export_manifest_are_consistent(tmp_path):
    store = Store(tmp_path)
    store.set_durations({"beef_rolls": 15})
    labels = list(SAMPLE_LABELS) * 2
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda label: store.add_sample(Image.new("RGBA", (20, 15), "red"), label, "meal-1"), labels
        ))
    assert len({row["sample_id"] for row in results}) == len(labels)
    assert store.summary()["counts"] == dict.fromkeys(SAMPLE_LABELS, 2)
    archive_path = store.export()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            rows = [json.loads(line) for line in archive.read("samples.jsonl").splitlines()]
            assert rows == store.records()
            assert set(archive.namelist()) == {"samples.jsonl", "config.json", *[row["path"] for row in rows]}
            assert json.loads(archive.read("config.json"))["durations"]["beef_rolls"] == 15
            for row in rows:
                with Image.open(io.BytesIO(archive.read(row["path"]))) as image:
                    assert image.format == "JPEG"
                    assert image.mode == "RGB"
                    assert image.size == (20, 15)
    finally:
        archive_path.unlink()


@pytest.mark.parametrize("label, session", [("unknown", "meal-1"), ("beef_rolls", "../escape"), ("background", ""), ("background", "x" * 81)])
def test_capture_rejects_invalid_labels_and_session_names(tmp_path, label, session):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        store.add_sample(Image.new("RGB", (8, 8)), label, session)
    assert store.records() == []
    assert list((tmp_path / "samples").iterdir()) == []


def test_export_rejects_a_manifest_path_outside_samples(tmp_path):
    store = Store(tmp_path / "data")
    secret = tmp_path / "outside.txt"
    secret.write_text("must not be exported")
    store.manifest.write_text(json.dumps({"label": "background", "session_id": "meal-1", "path": "../outside.txt"}) + "\n")
    with pytest.raises(ValueError, match="Invalid sample path"):
        store.export()


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_image_decode_rejects_empty_or_unreadable_files(payload):
    with pytest.raises(ValueError):
        decode_image(payload)


def test_image_decode_rejects_large_pixel_dimensions():
    image = Image.new("L", (4001, 4000))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with pytest.raises(ValueError, match="16 megapixels"):
        decode_image(buffer.getvalue())
