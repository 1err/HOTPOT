import base64
import io
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from hotpot.ingredients import SAMPLE_LABELS
from hotpot.openai_recognition import OpenAIRecognizer, OpenAIUsage, RecognitionError


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    monkeypatch.setattr("hotpot.credentials.DEFAULT_KEY_FILE", tmp_path / "missing-key")


def response(label="beef_rolls", *, uncertain=False, reason="Thin marbled beef rolls.",
             input_tokens=1000, output_tokens=100, **overrides):
    values = {
        "status": "completed", "output": [],
        "output_text": json.dumps({"label": label, "uncertain": uncertain, "reason": reason}),
        "usage": SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply = reply or response()
        self.error = error
        self.requests = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.reply


def image():
    return Image.new("RGB", (120, 80), "white")


def test_cloud_request_preserves_image_extent_and_uses_structured_nine_ingredient_selection():
    frame = Image.new("RGB", (1200, 600), "green")
    frame.paste("red", (0, 0, 120, 600))
    frame.paste("blue", (1080, 0, 1200, 600))
    client = FakeClient()
    result = OpenAIRecognizer(client=client).predict(frame)
    request = client.requests[0]
    content = request["input"][0]["content"][1]
    transmitted = Image.open(io.BytesIO(base64.b64decode(content["image_url"].split(",", 1)[1])))
    assert transmitted.size == (768, 384)
    assert transmitted.getpixel((10, 192))[0] > 200
    assert transmitted.getpixel((757, 192))[2] > 200
    assert content["detail"] == "high"
    schema = request["text"]["format"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["label"]["enum"] == list(SAMPLE_LABELS)
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "none"}
    assert request["model"] == "gpt-5.4"
    assert request["max_output_tokens"] == 160
    assert result["ingredient_id"] == result["predicted_id"] == "beef_rolls"
    assert result["confidence"] is None and result["scores"] == []


@pytest.mark.parametrize("label", SAMPLE_LABELS)
def test_all_canonical_labels_and_background_remain_distinguishable(label):
    result = OpenAIRecognizer(client=FakeClient(response(label))).predict(image())
    assert result["predicted_id"] == label
    assert result["ingredient_id"] == (None if label == "background" else label)
    assert result["uncertain"] is (label == "background")


def test_uncertain_ingredient_is_not_reported_as_a_confident_classification():
    result = OpenAIRecognizer(client=FakeClient(response("udon_noodles", uncertain=True))).predict(image())
    assert result["predicted_id"] == "udon_noodles"
    assert result["uncertain"] is True
    assert result["confidence"] is None


def test_usage_prices_each_selected_model_and_is_preserved_across_reload_and_switch():
    ledger = OpenAIUsage(request_limit=3)
    full = OpenAIRecognizer(client=FakeClient(), usage=ledger, min_interval_ms=0)
    mini = OpenAIRecognizer("gpt-5.4-mini", client=FakeClient(), usage=ledger, min_interval_ms=0)
    full.predict(image())
    full.reload()
    mini.predict(image())
    usage = ledger.status()
    assert usage["attempts"] == usage["completed"] == usage["reported_requests"] == 2
    assert usage["input_tokens"] == 2000 and usage["output_tokens"] == 200
    assert usage["estimated_usd"] == pytest.approx(0.0052)
    assert mini.status()["attempts_remaining"] == 1
    assert "not an account spending limit" in usage["price_note"]


def test_cooldown_blocks_a_second_request_without_charging_an_attempt(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("hotpot.openai_recognition.time.monotonic", lambda: clock[0])
    client = FakeClient()
    recognizer = OpenAIRecognizer(client=client)
    recognizer.predict(image())
    clock[0] += 0.1
    with pytest.raises(RecognitionError) as failure:
        recognizer.predict(image())
    assert failure.value.status_code == 429
    assert failure.value.retry_after_seconds == 3
    assert len(client.requests) == 1 and recognizer.usage.status()["attempts"] == 1
    clock[0] += 3
    recognizer.predict(image())
    assert len(client.requests) == 2


def test_request_limit_cannot_be_reset_by_reloading_or_changing_models():
    ledger = OpenAIUsage(request_limit=1)
    recognizer = OpenAIRecognizer(client=FakeClient(), usage=ledger, min_interval_ms=0)
    recognizer.predict(image())
    recognizer.reload()
    switched = OpenAIRecognizer("gpt-5.4-mini", client=FakeClient(), usage=ledger, min_interval_ms=0)
    with pytest.raises(RecognitionError, match="1-request") as failure:
        switched.predict(image())
    assert failure.value.status_code == 429
    assert ledger.status()["attempts"] == 1


@pytest.mark.parametrize("reply", [
    response(output_text=""), response(output_text="not json"),
    response("not_a_label"), response(uncertain=1), response(reason=" "),
    response(output_text='{"label":"soft_tofu","uncertain":false,"reason":"white","confidence":0.9}'),
    response(status="incomplete"),
    response(output=[SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="private detail")])]),
])
def test_invalid_refused_or_incomplete_responses_are_not_silently_accepted(reply):
    client = FakeClient(reply)
    recognizer = OpenAIRecognizer(client=client)
    with pytest.raises(RecognitionError):
        recognizer.predict(image())
    usage = recognizer.usage.status()
    assert len(client.requests) == 1
    assert usage["attempts"] == 1 and usage["completed"] == 0
    assert usage["estimated_usd"] == pytest.approx(0.004)


@pytest.mark.parametrize("status_code,expected", [
    (401, "API key"), (403, "API key"), (429, "credit limit"), (500, "connection"),
])
def test_provider_errors_are_sanitized_and_failed_attempts_are_counted(status_code, expected):
    error = RuntimeError("Do not disclose sk-secret-token or private upstream body")
    error.status_code = status_code
    client = FakeClient(error=error)
    recognizer = OpenAIRecognizer(client=client)
    with pytest.raises(RecognitionError) as failure:
        recognizer.predict(image())
    assert expected in str(failure.value)
    assert "sk-secret" not in str(failure.value)
    assert len(client.requests) == 1
    assert recognizer.usage.status()["attempts"] == 1
    assert recognizer.usage.status()["unreported_attempts"] == 1
    assert recognizer.usage.status()["estimated_usd"] == 0


def test_missing_key_is_unavailable_without_importing_sdk_or_sending_a_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", None)
    recognizer = OpenAIRecognizer()
    assert recognizer.status()["ready"] is False
    assert "OPENAI_API_KEY" in recognizer.status()["message"]
    with pytest.raises(RecognitionError):
        recognizer.predict(image())
    assert recognizer.usage.status()["attempts"] == 0


def test_missing_optional_sdk_explains_how_to_install_it(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-private-key")
    monkeypatch.setitem(sys.modules, "openai", None)
    status = OpenAIRecognizer().status()
    assert status["ready"] is False
    assert "requirements-vision.txt" in status["message"]
    assert "sk-private-key" not in json.dumps(status)


def test_sdk_is_configured_without_retries_and_key_stays_out_of_status(monkeypatch):
    constructed = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-private-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: constructed.append(kwargs) or FakeClient()))
    recognizer = OpenAIRecognizer()
    assert recognizer.status()["ready"] is True
    assert constructed[0] == {
        "api_key": "sk-private-key", "base_url": "https://api.openai.com/v1", "timeout": 30.0, "max_retries": 0,
    }
    assert "sk-private-key" not in json.dumps(recognizer.status())
    assert recognizer.usage.status()["attempts"] == 0


def test_secret_file_initializes_sdk_without_putting_key_into_environment(tmp_path, monkeypatch):
    import os

    private_file = tmp_path / "private-key"
    private_file.write_text("project-dummy-secret\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(private_file))
    constructed = []
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=lambda **kwargs: constructed.append(kwargs) or FakeClient(),
    ))
    recognizer = OpenAIRecognizer()
    assert recognizer.status()["ready"] is True
    assert constructed[0]["api_key"] == "project-dummy-secret"
    assert "OPENAI_API_KEY" not in os.environ
    assert "project-dummy-secret" not in json.dumps(recognizer.status())
    assert recognizer.usage.status()["attempts"] == 0


def test_invalid_secret_file_reports_safe_setup_error_without_creating_client(tmp_path, monkeypatch):
    private_file = tmp_path / "private-key"
    private_file.write_text("private-dummy-secret extra-token", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(private_file))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=lambda **kwargs: pytest.fail("Invalid key constructed an SDK client"),
    ))
    recognizer = OpenAIRecognizer()
    status = recognizer.status()
    assert status["ready"] is False
    assert "configure_key.py" in status["message"]
    assert "private-dummy-secret" not in json.dumps(status)
    assert recognizer.usage.status()["attempts"] == 0


@pytest.mark.parametrize("limit", [0, -1, True, 10001, "bad"])
def test_invalid_request_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="HOTPOT_OPENAI_REQUEST_LIMIT"):
        OpenAIUsage(request_limit=limit)
