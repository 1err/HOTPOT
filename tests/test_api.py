import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hotpot.app import MAX_UPLOAD, create_app
from hotpot.ingredients import DEFAULT_DURATIONS, INGREDIENT_IDS


class FakeRecognizer:
    def __init__(self, ready=True):
        self.ready = ready
        self.reloads = 0
        self.frames = []

    def status(self):
        return {"state": "ready" if self.ready else "unavailable", "mode": "test", "label": "Test model", "message": "Model not prepared"}

    def predict(self, frame):
        self.frames.append(frame)
        return {"ingredient_id": "beef_rolls", "confidence": 0.9, "uncertain": False}

    def reload(self):
        self.reloads += 1


@pytest.fixture
def client(tmp_path):
    recognizer = FakeRecognizer()
    app = create_app(tmp_path / "data", tmp_path / "models", recognizer=recognizer)
    with TestClient(app) as connection:
        yield connection, recognizer


def png():
    buffer = io.BytesIO()
    Image.new("RGBA", (24, 16), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def upload(payload=None):
    return {"image": ("frame.png", png() if payload is None else payload, "image/png")}


def test_config_lists_user_ingredients_and_persists_nullable_seconds(client):
    connection, _ = client
    config = connection.get("/api/config").json()
    assert [item["id"] for item in config["ingredients"]] == [
        "beef_rolls", "crab_sticks", "napa_cabbage", "lotus_root", "soft_tofu", "wide_noodles",
        "udon_noodles", "shiitake_mushrooms", "beef_meatballs",
    ]
    assert [item["name"] for item in config["ingredients"]] == [
        "肥牛卷", "蟹棒", "大白菜", "藕片", "嫩豆腐", "宽粉", "乌冬面", "香菇", "牛肉丸",
    ]
    assert {item["id"]: item["duration_seconds"] for item in config["ingredients"]} == DEFAULT_DURATIONS
    response = connection.put("/api/config", json={"durations": {"beef_rolls": 15, "crab_sticks": 180}})
    assert response.status_code == 200
    assert connection.put("/api/config", json={"durations": {"beef_rolls": None}}).status_code == 200
    saved = {item["id"]: item["duration_seconds"] for item in connection.get("/api/config").json()["ingredients"]}
    assert saved["beef_rolls"] is None
    assert saved["crab_sticks"] == 180


@pytest.mark.parametrize("label, initial_seconds", [
    ("udon_noodles", 180), ("shiitake_mushrooms", 300), ("beef_meatballs", 480),
])
def test_new_ingredient_durations_can_be_edited_and_cleared(client, label, initial_seconds):
    connection, _ = client
    config = connection.get("/api/config").json()
    assert next(item["duration_seconds"] for item in config["ingredients"] if item["id"] == label) == initial_seconds
    response = connection.put("/api/config", json={"durations": {label: initial_seconds + 15}})
    assert response.status_code == 200
    assert next(item["duration_seconds"] for item in response.json()["ingredients"] if item["id"] == label) == initial_seconds + 15
    assert connection.put("/api/config", json={"durations": {label: None}}).status_code == 200
    saved = connection.get("/api/config").json()
    assert next(item["duration_seconds"] for item in saved["ingredients"] if item["id"] == label) is None


@pytest.mark.parametrize("updates", [
    {"unknown": 10}, {"beef_rolls": 0}, {"beef_rolls": 7201},
    {"beef_rolls": 1.5}, {"beef_rolls": True}, {"beef_rolls": "10"},
])
def test_api_rejects_invalid_durations(client, updates):
    connection, _ = client
    assert connection.put("/api/config", json={"durations": updates}).status_code == 422


def test_recognition_decodes_real_image_and_reload_uses_runtime(client):
    connection, recognizer = client
    response = connection.post("/api/recognize", files=upload())
    assert response.status_code == 200
    assert response.json()["ingredient_id"] == "beef_rolls"
    assert recognizer.frames[0].mode == "RGB"
    assert recognizer.frames[0].size == (24, 16)
    assert connection.post("/api/model/reload").status_code == 200
    assert recognizer.reloads == 1


def test_model_unavailable_does_not_disable_config_or_capture(client):
    connection, recognizer = client
    recognizer.ready = False
    assert connection.post("/api/recognize", files=upload()).status_code == 503
    assert recognizer.frames == []
    assert connection.get("/api/health").json()["ok"] is True
    assert connection.get("/api/config").status_code == 200
    assert connection.post("/api/samples", files=upload(), data={"label": "background", "session_id": "meal-1"}).status_code == 200


@pytest.mark.parametrize("endpoint", ["/api/recognize", "/api/samples"])
def test_api_rejects_bad_and_oversized_uploads(client, endpoint):
    connection, _ = client
    data = {"label": "background", "session_id": "meal-1"}
    assert connection.post(endpoint, files=upload(b"bad bytes"), data=data).status_code == 422
    assert connection.post(endpoint, files=upload(b"x" * (MAX_UPLOAD + 1)), data=data).status_code == 413


def test_dataset_capture_and_download_match(client):
    connection, _ = client
    labels = [*INGREDIENT_IDS, "background"]
    for label in labels:
        response = connection.post("/api/samples", files=upload(), data={"label": label, "session_id": "meal-1"})
        assert response.status_code == 200
    summary = connection.get("/api/dataset").json()
    assert summary["total"] == len(labels)
    assert summary["counts"] == dict.fromkeys(labels, 1)
    response = connection.get("/api/dataset/export")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        rows = [json.loads(line) for line in archive.read("samples.jsonl").splitlines()]
        assert [row["label"] for row in rows] == labels
        assert set(archive.namelist()) == {"samples.jsonl", "config.json", *[row["path"] for row in rows]}


@pytest.mark.parametrize("label, session", [("unknown", "meal-1"), ("beef_rolls", "../escape")])
def test_sample_api_rejects_invalid_labels_and_sessions(client, label, session):
    connection, _ = client
    response = connection.post("/api/samples", files=upload(), data={"label": label, "session_id": session})
    assert response.status_code == 422
    assert connection.get("/api/dataset").json()["total"] == 0
