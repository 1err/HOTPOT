"""UI integration checks against an isolated real server and a model stub.

Run: .venv/bin/python tests/browser_checks.py
Requires Chromium and requirements-dev.txt. This tests browser behavior and image
transport, not ingredient recognition accuracy. No production data is modified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
import zipfile

from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["beef_rolls", "crab_sticks", "napa_cabbage", "lotus_root", "soft_tofu", "wide_noodles", "udon_noodles", "shiitake_mushrooms", "beef_meatballs", "background"]
NEW_INGREDIENTS = [
    ("udon_noodles", "乌冬面", 180),
    ("shiitake_mushrooms", "香菇", 300),
    ("beef_meatballs", "牛肉丸", 480),
]
STORAGE_KEY = "hotpot-lab-v1"

SERVER_CODE = """
from pathlib import Path
import sys
import uvicorn
from hotpot.app import create_app
from hotpot.ingredients import SAMPLE_LABELS
class UIModelStub:
    def status(self):
        return {"state": "ready", "mode": "test", "label": "UI integration stub (test only)", "message": "No recognition accuracy is tested."}
    def reload(self):
        pass
    def predict(self, image):
        assert image.mode == "RGB" and image.width > 0 and image.height > 0
        return {"ingredient_id": "beef_rolls", "uncertain": False, "confidence": .94, "inference_ms": 12,
                "scores": [{"id": label, "score": .94 if label == "beef_rolls" else .06 / (len(SAMPLE_LABELS) - 1)} for label in SAMPLE_LABELS]}
app = create_app(Path(sys.argv[1]) / "data", Path(sys.argv[1]) / "models", recognizer=UIModelStub())
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
"""


def wait_for_server(url, process):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Isolated test server exited before startup")
        try:
            with urlopen(url + "/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("Isolated test server did not start")


def saved_state(page):
    return page.evaluate("key => JSON.parse(localStorage.getItem(key))", STORAGE_KEY)


def config_seconds(page, url, label="beef_rolls"):
    config = page.request.get(url + "/api/config").json()
    return next(item["duration_seconds"] for item in config["ingredients"] if item["id"] == label)


def wait_for_seconds(page, url, seconds):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if config_seconds(page, url) == seconds:
            return
        page.wait_for_timeout(100)
    raise AssertionError(f"Duration was not saved to server: expected {seconds!r}")


def assert_guide_above_result(page):
    expect(page.locator("#crop-guide")).to_be_visible()
    expect(page.locator("#frame-result")).to_be_visible()
    guide = page.locator("#crop-guide").bounding_box()
    result = page.locator("#frame-result").bounding_box()
    preview = page.locator("#viewfinder").bounding_box()
    card = page.locator(".camera-card").bounding_box()
    assert guide["y"] + guide["height"] <= result["y"], f"Ingredient guide overlaps its result overlay: guide={guide}, result={result}"
    assert preview["x"] + preview["width"] <= card["x"] + card["width"], f"Preview is clipped by camera card: preview={preview}, card={card}"


def assert_uploaded_crop_follows_guide(page):
    # This image grows redder from top to bottom. The guide is deliberately
    # above center; a stale geometric-center crop would have red around 128.
    center_red = page.locator("#capture-canvas").evaluate("canvas => canvas.getContext('2d').getImageData(Math.floor(canvas.width/2), Math.floor(canvas.height/2), 1, 1).data[0]")
    assert 40 < center_red < 120, f"Captured crop does not follow the upper guide: center red={center_red}"


def run_checks(page, url, temp_root):
    errors = []
    predictions = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: predictions.append(request.url) if request.url.endswith("/api/recognize") else None)
    page.goto(url)
    expect(page.locator("#model-label")).to_have_text("UI integration stub (test only)")
    expect(page.locator("[data-ingredient-id]")).to_have_count(9)
    expect(page.locator("#ingredient-count")).to_have_text("9")
    assert page.locator("#sample-label option").evaluate_all("options => options.map(option => option.value)") == ["", *LABELS]

    # Each new configured ingredient has its own correctly timed batch and
    # persistent deadline. These exercise UI wiring, not visual recognition.
    new_timers = {}
    for label, chinese_name, duration in NEW_INGREDIENTS:
        page.locator(f'[data-ingredient-id="{label}"]').click()
        expect(page.locator("#selected-name")).to_have_text(chinese_name)
        expect(page.locator("#duration-input")).to_have_value(str(duration))
        assert config_seconds(page, url, label) == duration
        page.locator("#start-timer").click()
        new_timers[label] = saved_state(page)["timers"][label]
        assert new_timers[label]["duration"] == duration
        assert new_timers[label]["deadline"] - new_timers[label]["started_at"] == duration * 1000
    expect(page.locator(".timer-item")).to_have_count(3)
    page.reload()
    expect(page.locator(".timer-item")).to_have_count(3)
    for label, chinese_name, _ in NEW_INGREDIENTS:
        assert saved_state(page)["timers"][label] == new_timers[label]
        page.locator(".timer-item").filter(has_text=chinese_name).locator("button").click()
    expect(page.locator(".timer-item")).to_have_count(0)

    # Duration editing uses the actual API, including explicit clearing to null.
    beef = page.locator('[data-ingredient-id="beef_rolls"]')
    beef.click()
    expect(page.locator("#duration-input")).to_have_value("15")
    page.locator("#duration-input").fill("120")
    wait_for_seconds(page, url, 120)
    page.locator("#duration-input").fill("")
    wait_for_seconds(page, url, None)
    page.reload()
    beef.click()
    expect(page.locator("#duration-input")).to_have_value("")
    expect(page.locator("#start-timer")).to_be_disabled()
    page.locator("#duration-input").fill("120")
    wait_for_seconds(page, url, 120)

    # A repeated add/recognition must not change an existing batch's deadline.
    page.locator("#start-timer").click()
    timer = saved_state(page)["timers"]["beef_rolls"]
    expect(page.locator(".timer-item")).to_have_count(1)
    expect(page.locator("#start-timer")).to_be_disabled()
    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("Space")
    assert saved_state(page)["timers"]["beef_rolls"] == timer

    # Real getUserMedia with a browser-generated camera exercises continuous
    # polling and frame encoding. Predictions are intentionally a stub.
    page.locator("#start-camera").click()
    expect(page.locator("#source-badge")).to_have_text("CAMERA LIVE", timeout=10000)
    deadline = time.monotonic() + 8
    while len(predictions) < 3 and time.monotonic() < deadline:
        page.wait_for_timeout(100)
    assert len(predictions) >= 3, "Live recognition did not keep polling"
    assert saved_state(page)["timers"]["beef_rolls"] == timer
    assert_guide_above_result(page)

    # A correction persists while the model keeps predicting another class,
    # beyond the old temporary 3.5-second selection lock.
    page.locator('[data-ingredient-id="crab_sticks"]').click()
    prediction_count = len(predictions)
    page.wait_for_timeout(4200)
    assert len(predictions) >= prediction_count + 2
    expect(page.locator("#selected-name")).to_have_text("蟹棒")
    expect(page.locator("#prediction-name")).to_have_text("肥牛卷")
    page.locator("#start-timer").click()
    expect(page.locator(".timer-item")).to_have_count(2)
    expect(page.locator("#selected-name")).to_have_text("肥牛卷", timeout=5000)
    page.locator(".timer-item").filter(has_text="蟹棒").locator("button").click()
    assert saved_state(page)["timers"]["beef_rolls"] == timer
    page.locator("#stop-camera").click()
    expect(page.locator("#source-badge")).to_have_text("NO SOURCE")

    # Uploaded images and every training label, including background, reach
    # real persisted samples and a consistent downloadable ZIP.
    image = temp_root / "transport-test.png"
    gradient = Image.new("RGB", (200, 160))
    gradient.putdata([(round(y * 255 / 159), 60, 80) for y in range(160) for _ in range(200)])
    gradient.save(image)
    page.locator("#image-upload").set_input_files(str(image))
    expect(page.locator("#source-badge")).to_have_text("UPLOADED IMAGE")
    expect(page.locator("#recognize-once")).to_be_enabled()
    assert_uploaded_crop_follows_guide(page)
    page.locator("#session-id").fill("browser-check-session")
    for index, label in enumerate(LABELS, start=1):
        page.locator("#sample-label").select_option(label)
        page.locator("#capture-sample").click()
        expect(page.locator("#capture-sample")).to_be_enabled()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if page.request.get(url + "/api/dataset").json()["total"] == index:
                break
            page.wait_for_timeout(100)
        assert page.request.get(url + "/api/dataset").json()["total"] == index
    with page.expect_download() as download_info:
        page.locator("#export-dataset").click()
    exported = temp_root / "export.zip"
    download_info.value.save_as(exported)
    with zipfile.ZipFile(exported) as archive:
        rows = [json.loads(line) for line in archive.read("samples.jsonl").splitlines()]
        assert [row["label"] for row in rows] == LABELS
        assert all(row["session_id"] == "browser-check-session" for row in rows)
        assert all(row["path"] in archive.namelist() for row in rows)

    # Timers restore absolute deadlines after reload, and expired timers become
    # visible reminders without being started again or repeatedly notified.
    page.reload()
    expect(page.locator(".timer-item")).to_have_count(1)
    assert saved_state(page)["timers"]["beef_rolls"] == timer
    page.evaluate("""key => {
      const state = JSON.parse(localStorage.getItem(key));
      state.timers.beef_rolls = {started_at: Date.now() - 121000, deadline: Date.now() - 1000, duration: 120, notified: false};
      localStorage.setItem(key, JSON.stringify(state));
    }""", STORAGE_KEY)
    page.reload()
    expect(page.locator(".timer-item.due")).to_have_count(1)
    expect(page.locator(".countdown")).to_have_text("Time’s up")
    assert saved_state(page)["timers"]["beef_rolls"]["notified"] is True
    page.locator(".timer-item button").click()
    expect(page.locator(".timer-item")).to_have_count(0)
    assert saved_state(page)["timers"] == {}

    # The mobile layout fits the viewport without horizontal scrolling.
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator("#image-upload").set_input_files(str(image))
    expect(page.locator("#source-badge")).to_have_text("UPLOADED IMAGE")
    expect(page.locator("#frame-result-name")).to_have_text("肥牛卷")
    expect(page.locator("#recognize-once")).to_be_enabled()
    assert_guide_above_result(page)
    assert_uploaded_crop_follows_guide(page)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert errors == [], f"Browser errors: {errors}"


def main():
    executable = os.environ.get("CHROMIUM_PATH", "/usr/bin/chromium")
    with tempfile.TemporaryDirectory(prefix="hotpot-browser-check-") as directory:
        temp_root = Path(directory)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        with (temp_root / "server.log").open("w+") as logfile:
            process = subprocess.Popen([sys.executable, "-c", SERVER_CODE, str(temp_root), str(port)], cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT)
            try:
                wait_for_server(url, process)
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(executable_path=executable, headless=True, args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
                    context = browser.new_context(permissions=["camera"], viewport={"width": 1440, "height": 1100}, accept_downloads=True)
                    run_checks(context.new_page(), url, temp_root)
                    browser.close()
                print("Browser integration passed: nine ingredients and configured new timers, durations, duplicate prevention, continuous webcam frames, persistent manual correction, ten capture labels/export, persisted and delayed timers, desktop/mobile guide layout. Model accuracy was not tested.")
            except Exception:
                logfile.seek(0)
                print(logfile.read(), file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
