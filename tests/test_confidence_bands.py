"""Qualitative GPT estimates remain distinct from measured probabilities.

All responses are mocked. These checks make no OpenAI requests and do not read
credentials; browser coverage runs when Playwright and a Chromium executable
are available.
"""

import io
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from PIL import Image

from hotpot.ingredients import DEFAULT_DURATIONS, INGREDIENTS
from hotpot.openai_recognition import OpenAIRecognizer, RecognitionError


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.requests = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(
            status="completed", output=[], output_text=json.dumps(self.result),
            usage=SimpleNamespace(input_tokens=1000, output_tokens=40),
        )


def predict(result):
    client = FakeClient(result)
    recognizer = OpenAIRecognizer(client=client, min_interval_ms=0)
    return recognizer.predict(Image.new("RGB", (120, 80), "white")), client


@pytest.mark.parametrize("level,uncertain", [("high", False), ("medium", False), ("low", True)])
def test_band_is_model_self_assessment_and_low_forces_uncertainty(level, uncertain):
    result, client = predict({
        "label": "beef_rolls", "uncertain": False,
        "reason": "Thin rolled beef slices.", "confidence_level": level,
    })
    assert result["confidence_level"] == level
    assert result["confidence_basis"] == "model_self_assessment"
    assert result["uncertain"] is uncertain
    assert result["ingredient_id"] == "beef_rolls"
    assert result["confidence"] is None and result["scores"] == []
    schema = client.requests[0]["text"]["format"]["schema"]
    assert "confidence_level" in schema["required"]
    assert schema["properties"]["confidence_level"]["enum"] == ["high", "medium", "low"]
    assert "not calibrated probabilities" in client.requests[0]["instructions"]


@pytest.mark.parametrize("level", ["high", "medium", "low"])
def test_band_does_not_override_explicit_uncertainty(level):
    result, _ = predict({
        "label": "udon_noodles", "uncertain": True,
        "reason": "Only part of the noodle is visible.", "confidence_level": level,
    })
    assert result["uncertain"] is True
    assert result["confidence_level"] == level


def test_background_remains_uncertain_even_with_high_band():
    result, _ = predict({
        "label": "background", "uncertain": False,
        "reason": "An empty plate is clearly visible.", "confidence_level": "high",
    })
    assert result["ingredient_id"] is None
    assert result["uncertain"] is True


@pytest.mark.parametrize("level", [None, "", "HIGH", "certain", 0.9, True, [], {}])
def test_invalid_confidence_band_is_rejected_without_counting_completion(level):
    client = FakeClient({
        "label": "beef_rolls", "uncertain": False,
        "reason": "Thin rolled beef slices.", "confidence_level": level,
    })
    recognizer = OpenAIRecognizer(client=client)
    with pytest.raises(RecognitionError, match="invalid ingredient result"):
        recognizer.predict(Image.new("RGB", (120, 80), "white"))
    assert recognizer.usage.status()["attempts"] == 1
    assert recognizer.usage.status()["completed"] == 0


def test_extra_fields_still_rejected_with_valid_band():
    with pytest.raises(RecognitionError, match="invalid ingredient result"):
        predict({
            "label": "beef_rolls", "uncertain": False,
            "reason": "Thin rolled beef slices.", "confidence_level": "high",
            "probability": 0.99,
        })


@pytest.mark.parametrize("uncertain", [False, True])
def test_legacy_response_does_not_invent_a_confidence_level(uncertain):
    result, _ = predict({
        "label": "beef_rolls", "uncertain": uncertain,
        "reason": "Thin rolled beef slices.",
    })
    assert "confidence_level" not in result and "confidence_basis" not in result
    assert result["uncertain"] is uncertain
    assert result["confidence"] is None and result["scores"] == []


def test_browser_displays_estimates_without_probabilities_and_preserves_local_scores():
    playwright = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("CHROMIUM_PATH") or shutil.which("chromium") or shutil.which("google-chrome")
    if not executable:
        pytest.skip("Chromium executable is not installed")
    root = Path(__file__).resolve().parents[1]
    source_image = io.BytesIO()
    Image.new("RGB", (800, 600), "red").save(source_image, "PNG")
    errors = []
    cases = [("high", "HIGH · GPT ESTIMATE"), ("medium", "MEDIUM · GPT ESTIMATE"),
             ("low", "LOW · GPT ESTIMATE"), (None, "RECOGNIZED"), ("local", "0.940 MATCH SCORE")]
    with playwright.sync_playwright() as engine:
        browser = engine.chromium.launch(executable_path=executable, headless=True)
        try:
            for level, badge in cases:
                context = browser.new_context(viewport={"width": 390, "height": 844})
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                cloud = level != "local"
                model = {"state": "ready", "provider": "openai" if cloud else "local", "cloud": cloud,
                         "model": "gpt-5.4" if cloud else "local", "label": "Test recognizer"}
                prediction = {"ingredient_id": "beef_rolls", "uncertain": False, "scores": [],
                              "confidence": None, "reason": "Thin rolled beef slices."}
                if level in ("high", "medium", "low"):
                    # Also exercise the browser's low-band guard independently
                    # of the backend setting uncertain=true.
                    prediction.update(confidence_level=level, confidence_basis="model_self_assessment")
                elif not cloud:
                    prediction.update(confidence=0.94, scores=[{"id": "beef_rolls", "score": 0.94}])

                def route_request(route):
                    path = urlparse(route.request.url).path
                    if path == "/":
                        route.fulfill(content_type="text/html", body=(root / "static/index.html").read_text())
                    elif path.startswith("/static/"):
                        suffix = Path(path).suffix
                        route.fulfill(content_type="text/javascript" if suffix == ".js" else "text/css",
                                      body=(root / path.lstrip("/")).read_text())
                    elif path == "/api/config":
                        route.fulfill(json={"ingredients": [{**item, "duration_seconds": DEFAULT_DURATIONS[item["id"]]}
                                                             for item in INGREDIENTS], "session_id": "confidence-ui"})
                    elif path == "/api/model":
                        route.fulfill(json=model)
                    elif path == "/api/dataset":
                        route.fulfill(json={"counts": {}, "total": 0})
                    elif path == "/api/recognize":
                        route.fulfill(json=prediction)
                    else:
                        route.fulfill(status=404)

                page.route("**/*", route_request)
                page.goto("http://hotpot.test/")
                playwright.expect(page.locator("#recognition-model")).to_be_enabled()
                page.locator("#image-upload").set_input_files({
                    "name": "ingredient.png", "mimeType": "image/png", "buffer": source_image.getvalue(),
                })
                if cloud:
                    playwright.expect(page.locator("#recognize-once")).to_be_enabled()
                    page.locator("#recognize-once").click()
                playwright.expect(page.locator("#confidence-badge")).to_have_text(badge)
                if level in ("high", "medium", "low"):
                    playwright.expect(page.locator("#confidence-note")).to_be_visible()
                    playwright.expect(page.locator("#confidence-note")).to_contain_text("not a validated probability")
                    playwright.expect(page.locator("#frame-result-score")).to_have_text(f"{level.upper()} GPT estimate")
                    assert "%" not in page.locator("#confidence-badge").inner_text()
                else:
                    playwright.expect(page.locator("#confidence-note")).to_be_hidden()
                if level == "low":
                    playwright.expect(page.locator("#selected-name")).to_have_text("Choose an ingredient")
                    playwright.expect(page.locator("#frame-result-state")).to_contain_text("AMBIGUOUS")
                else:
                    playwright.expect(page.locator("#selected-name")).to_have_text("肥牛卷")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                context.close()
        finally:
            browser.close()
    assert errors == []
