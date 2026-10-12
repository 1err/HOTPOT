"""Benchmark provenance, preflight and paid-resume behavior using mocked pixels/models."""
from copy import deepcopy
import fcntl
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request

from PIL import Image
import pytest

from hotpot.ingredients import INGREDIENT_IDS
from hotpot.openai_recognition import RecognitionError
from scripts.benchmark_public_images import (
    ROOT, SourceRedirects, digest, load_manifest, plan, prepare_image, run_benchmark,
    stage_images,
)


def sources():
    return [{"id": f"photo-{index}", "label": label,
             "image_url": f"https://upload.wikimedia.org/wikipedia/commons/photo-{index}.jpg",
             "page_url": f"https://commons.wikimedia.org/wiki/File:Photo-{index}.jpg",
             "description": f"Ingredient photograph {index}", "author": "Example Photographer",
             "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
             "review_status": "accepted"}
            for index, label in enumerate(label for label in INGREDIENT_IDS for _ in range(3))]


def write_manifest(path, rows):
    path.write_text(json.dumps({"schema_version": 1, "sources": rows}))
    return path


def photo_bytes(index, size=(64, 32)):
    image = Image.new("RGB", size, (index * 7, index * 3, index))
    exif = Image.Exif()
    exif[270] = "Private metadata must not be sent"
    result = BytesIO()
    image.save(result, "PNG", exif=exif)
    return result.getvalue()


def fetch_mock(url):
    index = int(url.rsplit("photo-", 1)[1].split(".")[0])
    return photo_bytes(index)


class FakeRecognizer:
    def __init__(self, output, *, ready=True, fail_index=None, interrupt_index=None):
        self.output = output
        self.ready = ready
        self.fail_index = fail_index
        self.interrupt_index = interrupt_index
        self.calls = []
        self.usage = {"attempts": 0, "completed": 0, "input_tokens": 0,
                      "output_tokens": 0, "estimated_usd": 0, "unreported_attempts": 0}

    def status(self):
        return {"ready": self.ready, "usage": dict(self.usage)}

    def predict(self, image):
        # The input is pixels only: no filename, label, or source URL reaches the model.
        assert not image.info
        index = len(self.calls)
        self.calls.append(image.copy())
        stored = json.loads((self.output / "results-gpt-5.4.json").read_text())
        assert stored["predictions"][-1]["status"] == "started"
        self.usage["attempts"] += 1
        self.usage["input_tokens"] += 100
        self.usage["output_tokens"] += 20
        self.usage["estimated_usd"] += .001
        if index == self.interrupt_index:
            raise KeyboardInterrupt()
        if index == self.fail_index:
            raise RecognitionError("Mock request failed; no automatic retry.")
        self.usage["completed"] += 1
        source_index = image.getpixel((0, 0))[0] // 7
        label = sources()[source_index]["label"]
        return {"predicted_id": label if source_index != 1 else "background",
                "uncertain": source_index in (1, 2), "confidence_level": "high",
                "reason": "Visible ingredient features"}


def test_default_cli_is_plan_only_without_network_secret_or_output(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.json", sources())
    output = tmp_path / "no-output"
    result = subprocess.run([sys.executable, "scripts/benchmark_public_images.py", "--manifest", str(manifest),
                             "--output", str(output)], cwd=ROOT, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["planned_api_calls"] == 27 and report["mode"] == "plan"
    assert all(row["status"] == "not_run" for row in report["results"])
    assert not output.exists()


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows[0].update(label="background"),
    lambda rows: rows[0].update(label="crab_sticks"),
    lambda rows: rows[0].update(id="../../secret"),
    lambda rows: rows[0].update(image_url="https://127.0.0.1/private"),
    lambda rows: rows[0].update(image_url="http://upload.wikimedia.org/photo.jpg"),
    lambda rows: rows[0].update(image_url=rows[1]["image_url"]),
    lambda rows: rows[0].update(crop_box=[0, .5, 1, .4]),
])
def test_manifest_rejects_invalid_benchmark_or_unsafe_download(tmp_path, change):
    rows = sources()
    change(rows)
    with pytest.raises(ValueError):
        load_manifest(write_manifest(tmp_path / "manifest.json", rows))


def test_source_redirect_cannot_reach_local_or_unselected_host():
    with pytest.raises(ValueError):
        SourceRedirects().redirect_request(Request(sources()[0]["image_url"]), None, 302, "Found", {},
                                          "https://169.254.169.254/latest/meta-data")


def test_stage_preserves_exact_sources_and_request_pixels_with_resumable_cache(tmp_path):
    records = stage_images(sources(), tmp_path, fetch=fetch_mock)
    assert len(records) == 27 and (tmp_path / "gallery.html").exists()
    first = records[0]
    assert first["source_sha256"] == digest(photo_bytes(0))
    assert first["license"] == "CC-BY-SA-4.0" and first["page_url"] == sources()[0]["page_url"]
    with Image.open(tmp_path / first["input_path"]) as image:
        assert image.mode == "RGB" and not image.info
        jpeg = BytesIO()
        image.save(jpeg, "JPEG", quality=90)
        assert first["request_image_sha256"] == digest(jpeg.getvalue())
    repeated = stage_images(sources(), tmp_path, fetch=lambda _: pytest.fail("Cached image must not redownload"))
    assert repeated == records
    assert not (tmp_path / "samples.jsonl").exists()


def test_invalid_or_duplicate_pixels_never_count_as_distinct_photos(tmp_path):
    with pytest.raises(ValueError, match="decode"):
        stage_images(sources(), tmp_path / "bad", fetch=lambda _: b"not an image")
    with pytest.raises(ValueError, match="Duplicate decoded"):
        stage_images(sources(), tmp_path / "duplicates", fetch=lambda _: photo_bytes(1))


def test_processed_image_is_lossless_cropped_and_matches_app_resolution():
    processed, metadata = prepare_image(photo_bytes(4, (1600, 800)), [.25, 0, .75, 1])
    with Image.open(BytesIO(processed)) as image:
        assert image.size == (768, 768) and not image.info
    assert metadata["original_size"] == [1600, 800]


def test_run_reports_every_image_and_resumes_without_billable_repeats(tmp_path):
    records = stage_images(sources(), tmp_path, fetch=fetch_mock)
    fake = FakeRecognizer(tmp_path, fail_index=5)
    report = run_benchmark(records, tmp_path, "gpt-5.4", recognizer=fake)
    assert len(fake.calls) == 6 and len(report["predictions"]) == 27
    assert report["summary"]["completed"] == 5 and report["summary"]["correct"] == 4
    assert report["summary"]["errors"] == 1 and report["summary"]["accuracy"] == 4 / 5
    assert report["summary"]["not_run"] == 21
    assert report["summary"]["estimated_usd"] == pytest.approx(.006)
    assert report["predictions"][1]["predicted_label"] == "background"
    assert report["predictions"][5]["error"] == "Mock request failed; no automatic retry."
    assert report["predictions"][1]["confidence_level"] == "high"
    assert "confidence" not in report["predictions"][0]
    assert (tmp_path / "results-gpt-5.4.md").exists()
    second = FakeRecognizer(tmp_path)
    repeated = run_benchmark(records, tmp_path, "gpt-5.4", recognizer=second)
    assert len(second.calls) == 21 and repeated["summary"]["completed"] == 26
    assert repeated["summary"]["correct"] == 25
    third = FakeRecognizer(tmp_path, ready=False)
    cached = run_benchmark(records, tmp_path, "gpt-5.4", recognizer=third)
    assert cached == repeated and third.calls == []


def test_all_pixels_and_review_status_preflight_before_first_paid_call(tmp_path):
    records = stage_images(sources(), tmp_path, fetch=fetch_mock)
    fake = FakeRecognizer(tmp_path)
    unreviewed = deepcopy(records)
    unreviewed[-1]["review_status"] = "unreviewed"
    with pytest.raises(ValueError, match="Visually review"):
        run_benchmark(unreviewed, tmp_path, "gpt-5.4", recognizer=fake)
    (tmp_path / records[-1]["input_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        run_benchmark(records, tmp_path, "gpt-5.4", recognizer=fake)
    assert fake.calls == [] and not (tmp_path / "results-gpt-5.4.json").exists()


def test_missing_credentials_does_not_make_or_record_attempts(tmp_path):
    records = stage_images(sources(), tmp_path, fetch=fetch_mock)
    fake = FakeRecognizer(tmp_path, ready=False)
    with pytest.raises(ValueError, match="private OPENAI_API_KEY"):
        run_benchmark(records, tmp_path, "gpt-5.4", recognizer=fake)
    assert fake.calls == [] and not (tmp_path / "results-gpt-5.4.json").exists()


def test_interrupted_attempt_is_preserved_and_not_automatically_retried(tmp_path):
    records = stage_images(sources(), tmp_path, fetch=fetch_mock)
    fake = FakeRecognizer(tmp_path, interrupt_index=0)
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(records, tmp_path, "gpt-5.4", recognizer=fake)
    stored = json.loads((tmp_path / "results-gpt-5.4.json").read_text())
    assert stored["predictions"][0]["status"] == "started"
    second = FakeRecognizer(tmp_path)
    report = run_benchmark(records, tmp_path, "gpt-5.4", recognizer=second)
    assert len(second.calls) == 26 and len(report["predictions"]) == 27
    assert report["predictions"][0]["status"] == "started"


def test_explicit_rerun_archives_prior_results(tmp_path):
    records = stage_images(sources(), tmp_path, fetch=fetch_mock)
    report = run_benchmark(records, tmp_path, "gpt-5.4", recognizer=FakeRecognizer(tmp_path))
    second = FakeRecognizer(tmp_path)
    run_benchmark(records, tmp_path, "gpt-5.4", recognizer=second, rerun=True)
    archived = list(tmp_path.glob("results-gpt-5.4-*.json"))
    assert len(second.calls) == 27 and len(archived) == 1
    assert json.loads(archived[0].read_text()) == report


def test_simultaneous_run_cannot_repeat_paid_requests(tmp_path):
    records = stage_images(sources(), tmp_path, fetch=fetch_mock)
    fake = FakeRecognizer(tmp_path)
    with (tmp_path / ".benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another benchmark"):
            run_benchmark(records, tmp_path, "gpt-5.4", recognizer=fake)
    assert fake.calls == []


def test_modified_download_cache_is_preserved_not_overwritten(tmp_path):
    records = stage_images(sources(), tmp_path, fetch=fetch_mock)
    source = tmp_path / records[-1]["source_path"]
    source.write_bytes(b"modified source")
    with pytest.raises(ValueError, match="modified"):
        stage_images(sources(), tmp_path, fetch=lambda _: pytest.fail("No network for modified cache"))
    assert source.read_bytes() == b"modified source"
