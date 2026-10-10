import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hotpot.app import create_app
from hotpot.openai_recognition import OpenAIRecognizer


class LocalRecognizer:
    def status(self):
        return {"state": "ready", "mode": "test", "label": "Local test", "message": "Local test model"}

    def predict(self, frame):
        return {"ingredient_id": "soft_tofu", "uncertain": False}


class CloudClient:
    def __init__(self):
        self.requests = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            status="completed", output=[],
            output_text=json.dumps({"label": "lotus_root", "uncertain": False, "reason": "Round slices with holes."}),
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        )


def upload():
    buffer = io.BytesIO()
    Image.new("RGB", (900, 500), "white").save(buffer, "PNG")
    return {"image": ("camera.png", buffer.getvalue(), "image/png")}


def test_explicit_cloud_selection_without_key_is_reviewable_and_does_not_disable_timers(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(tmp_path / "data", tmp_path / "models", recognizer=LocalRecognizer())
    with TestClient(app) as client:
        assert client.get("/api/model").json()["provider"] == "local"
        selected = client.put("/api/model/provider", json={"provider": "openai", "model": "gpt-5.4-mini"})
        assert selected.status_code == 200
        status = selected.json()
        assert status["provider"] == "openai" and status["model"] == "gpt-5.4-mini"
        assert status["ready"] is False and status["cloud"] is True
        assert status["min_interval_ms"] == 3000 and status["recommended_manual"] is True
        assert client.post("/api/recognize", files=upload()).status_code == 503
        assert client.get("/api/config").status_code == 200
        assert client.put("/api/config", json={"durations": {"udon_noodles": 180}}).status_code == 200
        assert client.post("/api/model/reload").json()["usage"]["attempts"] == 0


@pytest.mark.parametrize("selection", [
    {"provider": "other"}, {"provider": "openai", "model": "unapproved-model"},
    {"provider": "openai", "api_key": "must-not-accept-browser-secret"},
])
def test_provider_api_rejects_invalid_selection_and_credentials_in_payload(tmp_path, selection):
    app = create_app(tmp_path / "data", tmp_path / "models", recognizer=LocalRecognizer())
    with TestClient(app) as client:
        assert client.put("/api/model/provider", json=selection).status_code == 422


def test_cloud_cooldown_is_exposed_as_http_retry_after_and_usage_survives_model_switch(tmp_path, monkeypatch):
    cloud = CloudClient()
    monkeypatch.setattr("hotpot.openai_recognition.OpenAIRecognizer", lambda model, usage: OpenAIRecognizer(model, client=cloud, usage=usage))
    app = create_app(tmp_path / "data", tmp_path / "models", recognizer=LocalRecognizer())
    with TestClient(app) as client:
        client.put("/api/model/provider", json={"provider": "openai", "model": "gpt-5.4"})
        recognized = client.post("/api/recognize", files=upload())
        assert recognized.status_code == 200
        assert recognized.json()["ingredient_id"] == "lotus_root"
        repeated = client.post("/api/recognize", files=upload())
        assert repeated.status_code == 429
        assert 1 <= int(repeated.headers["Retry-After"]) <= 3
        switched = client.put("/api/model/provider", json={"provider": "openai", "model": "gpt-5.4-mini"}).json()
        assert switched["usage"]["attempts"] == 1
        assert switched["usage"]["estimated_usd"] == pytest.approx(0.004)
        assert switched["attempts_remaining"] == 99
        assert len(cloud.requests) == 1


def test_request_limit_and_usage_remain_in_effect_after_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTPOT_OPENAI_REQUEST_LIMIT", "1")
    monkeypatch.setattr("hotpot.openai_recognition.OpenAIRecognizer", lambda model, usage: OpenAIRecognizer(model, client=CloudClient(), usage=usage, min_interval_ms=0))
    app = create_app(tmp_path / "data", tmp_path / "models", recognizer=LocalRecognizer())
    with TestClient(app) as client:
        client.put("/api/model/provider", json={"provider": "openai"})
        assert client.post("/api/recognize", files=upload()).status_code == 200
        assert client.post("/api/model/reload").json()["attempts_remaining"] == 0
        exhausted = client.post("/api/recognize", files=upload())
        assert exhausted.status_code == 429 and "1-request" in exhausted.json()["detail"]


def test_cloud_startup_does_not_import_local_ml_dependencies(tmp_path):
    script = """
import builtins
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.startswith(('torch', 'open_clip')) or name in ('training', 'recognition'):
        raise AssertionError('Cloud startup imported local ML: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from hotpot.app import create_app
from fastapi.testclient import TestClient
with TestClient(create_app()) as client:
    status = client.get('/api/model').json()
    assert status['provider'] == 'openai' and not status['ready']
    assert 'OPENAI_API_KEY' in status['message']
print('cloud-only startup passed')
"""
    environment = dict(os.environ)
    environment.pop("OPENAI_API_KEY", None)
    environment.update({
        "HOTPOT_RECOGNITION_PROVIDER": "openai", "HOTPOT_DATA_DIR": str(tmp_path / "data"),
        "HOTPOT_MODEL_DIR": str(tmp_path / "models"),
    })
    result = subprocess.run([sys.executable, "-c", script], env=environment, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "cloud-only startup passed" in result.stdout
