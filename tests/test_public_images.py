"""Public-image staging checks use explicit mock image bytes, never live downloads."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from hotpot.ingredients import INGREDIENT_IDS
from scripts.prepare_public_images import (
    DEFAULT_MANIFEST, MAX_DOWNLOAD_BYTES, MAX_SOURCE_PIXELS, ImageHostRedirects,
    download_sources, load_sources, normalize_image, summarize, validate_url,
)


def photo_bytes(size=(80, 40)):
    image = Image.new("RGB", size, "green")
    exif = Image.Exif()
    exif[270] = "Private metadata should be removed"
    output = BytesIO()
    image.save(output, "JPEG", exif=exif)
    return output.getvalue()


def write_manifest(tmp_path, sources):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"schema_version": 1, "sources": sources}))
    return path


def test_source_contract_covers_all_nine_without_claiming_training():
    sources = load_sources()
    summary = summarize(sources)
    assert len(sources) == 35
    assert all(summary["counts"][label] > 0 for label in INGREDIENT_IDS)
    assert summary["counts"]["background"] == 4
    assert all(source["review_status"] == "unreviewed" for source in sources)
    assert summary["training_ready"] is False
    # A photographer is kept together even when their images span food classes.
    fumikas = [source for source in sources if source["author"] == "Fumikas Sagisavas"]
    assert len({source["source_group"] for source in fumikas}) == 1


def test_dry_run_has_no_network_or_output_writes(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run([
        sys.executable, "scripts/prepare_public_images.py", "--dry-run", "--labels", "udon_noodles",
        "--output", str(output),
    ], cwd=DEFAULT_MANIFEST.parents[1], capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout)
    assert summary["candidate_images"] == 2
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("id", "../escape"), ("source_group", "../../outside"), ("label", "mystery-food"),
    ("author", ""), ("license", "unknown"), ("image_url", "https://127.0.0.1/image.jpg"),
    ("review_status", "trained"),
])
def test_source_validation_rejects_missing_metadata_and_unsafe_paths(tmp_path, field, value):
    sources = deepcopy(load_sources()[:1])
    sources[0][field] = value
    with pytest.raises(ValueError):
        load_sources(write_manifest(tmp_path, sources))


def test_duplicate_photos_are_not_counted_twice(tmp_path):
    sources = deepcopy(load_sources()[:1])
    duplicate = deepcopy(sources[0])
    duplicate["id"] = "a-different-name"
    with pytest.raises(ValueError, match="Duplicate"):
        load_sources(write_manifest(tmp_path, sources + [duplicate]))


def test_staging_preserves_provenance_and_group_without_import_or_training(tmp_path):
    sources = load_sources()[:1]
    raw = photo_bytes((2400, 1200))
    result = download_sources(sources, tmp_path, fetch=lambda _: raw)
    assert result["downloaded"] == 1 and result["imported"] == 0 and result["trained"] is False
    record = json.loads((tmp_path / "provenance.jsonl").read_text())
    assert record["source_group"] == sources[0]["source_group"]
    assert record["author"] == sources[0]["author"]
    assert record["license_url"] == sources[0]["license_url"]
    assert record["review_status"] == "unreviewed"
    assert record["saved_size"] == [1600, 800]
    with Image.open(tmp_path / record["saved_path"]) as image:
        assert not image.getexif()
    assert (tmp_path / "gallery.html").exists()
    assert not (tmp_path / "samples.jsonl").exists()
    assert not (tmp_path / "models").exists()
    repeated = download_sources(sources, tmp_path, fetch=lambda _: pytest.fail("Already downloaded"))
    assert repeated["already_present"] == 1 and not repeated["failures"]


def test_invalid_bytes_do_not_create_a_labeled_photo(tmp_path):
    result = download_sources(load_sources()[:1], tmp_path, fetch=lambda _: b"not a photo")
    assert result["downloaded"] == 0 and len(result["failures"]) == 1
    assert not list(tmp_path.rglob("*.jpg"))
    record = json.loads((tmp_path / "provenance.jsonl").read_text())
    assert record["download_status"] == "failed" and "saved_path" not in record


def test_download_limits_and_dimensions_are_checked(monkeypatch):
    with pytest.raises(ValueError, match="20 MiB"):
        normalize_image(b"x" * (MAX_DOWNLOAD_BYTES + 1))
    monkeypatch.setattr("scripts.prepare_public_images.MAX_SOURCE_PIXELS", 100)
    with pytest.raises(ValueError, match="megapixel"):
        normalize_image(photo_bytes((20, 20)))


def test_redirects_cannot_leave_the_image_host():
    from urllib.request import Request
    with pytest.raises(ValueError, match="upload.wikimedia.org"):
        ImageHostRedirects().redirect_request(
            Request(load_sources()[0]["image_url"]), None, 302, "Found", {}, "https://example.com/image.jpg",
        )
    with pytest.raises(ValueError):
        validate_url("http://upload.wikimedia.org/image.jpg", "upload.wikimedia.org")


def test_modified_local_review_image_is_not_overwritten(tmp_path):
    sources = load_sources()[:1]
    download_sources(sources, tmp_path, fetch=lambda _: photo_bytes())
    record = json.loads((tmp_path / "provenance.jsonl").read_text())
    destination = tmp_path / record["saved_path"]
    modified = photo_bytes((10, 10))
    destination.write_bytes(modified)
    result = download_sources(sources, tmp_path, fetch=lambda _: pytest.fail("Do not replace reviewed images"))
    assert len(result["failures"]) == 1
    assert destination.read_bytes() == modified
