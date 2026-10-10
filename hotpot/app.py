"""Local FastAPI app. Camera frames arrive from the user's browser."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictInt
from starlette.background import BackgroundTask

from .ingredients import INGREDIENTS
from .storage import Store, decode_image

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 6 * 1024 * 1024
logger = logging.getLogger(__name__)


class ConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    durations: dict[str, StrictInt | None]


class ProviderUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["local", "openai"]
    model: Literal["gpt-5.4", "gpt-5.4-mini"] | None = None


def create_app(data_dir: Path | None = None, model_dir: Path | None = None, recognizer=None) -> FastAPI:
    store = Store(data_dir or Path(os.environ.get("HOTPOT_DATA_DIR", ROOT / "data")))
    model_path = model_dir or Path(os.environ.get("HOTPOT_MODEL_DIR", ROOT / "models"))
    model_path.mkdir(parents=True, exist_ok=True)
    inference_lock = asyncio.Lock()
    capture_session = f"meal-{uuid.uuid4().hex[:8]}"

    async def make_recognizer(provider, model=None):
        if provider == "openai":
            from .openai_recognition import OpenAIRecognizer, OpenAIUsage
            if app.state.openai_usage is None:
                app.state.openai_usage = OpenAIUsage()
            return await asyncio.to_thread(
                OpenAIRecognizer, model or app.state.openai_model, usage=app.state.openai_usage,
            )
        if provider != "local":
            raise ValueError("HOTPOT_RECOGNITION_PROVIDER must be local or openai.")
        from .recognition import Recognizer
        return await asyncio.to_thread(Recognizer, store.root, model_path)

    @asynccontextmanager
    async def lifespan(app):
        if app.state.recognizer is None:
            try:
                app.state.recognizer = await make_recognizer(app.state.provider)
            except Exception as error:
                if app.state.provider == "openai":
                    logger.warning("Cloud recognition runtime could not be initialized")
                    app.state.model_error = "Cloud recognition could not be initialized. Check HOTPOT_OPENAI_MODEL, HOTPOT_OPENAI_REQUEST_LIMIT and requirements-vision.txt."
                else:
                    logger.exception("Recognition runtime could not be initialized")
                    app.state.model_error = str(error)
        try:
            yield
        finally:
            async with inference_lock:
                close = getattr(app.state.recognizer, "close", None)
                if close is not None:
                    try:
                        await asyncio.to_thread(close)
                    except Exception:
                        logger.warning("Recognition client cleanup failed")

    app = FastAPI(title="Hotpot Lab", version="0.3.0", lifespan=lifespan)
    app.state.recognizer = recognizer
    app.state.model_error = None
    app.state.store = store
    app.state.provider = "local" if recognizer is not None else os.environ.get("HOTPOT_RECOGNITION_PROVIDER", "local").strip().lower()
    app.state.openai_model = os.environ.get("HOTPOT_OPENAI_MODEL", "gpt-5.4").strip()
    app.state.openai_usage = None

    def model_status():
        if app.state.recognizer is not None:
            status = dict(app.state.recognizer.status())
            status.setdefault("provider", app.state.provider)
            status.setdefault("cloud", app.state.provider == "openai")
            status.setdefault("recommended_manual", False)
            status.setdefault("min_interval_ms", 750)
            return status
        return {
            "state": "unavailable", "mode": "unavailable", "label": "Recognition not prepared",
            "ready": False, "provider": app.state.provider, "cloud": app.state.provider == "openai",
            "model": app.state.openai_model if app.state.provider == "openai" else None,
            "recommended_manual": app.state.provider == "openai", "min_interval_ms": 3000 if app.state.provider == "openai" else 750,
            "message": app.state.model_error or "Prepare the model to enable recognition. Timers and capture work independently.",
        }

    def config():
        durations = store.durations()
        return {
            "ingredients": [{**item, "duration_seconds": durations[item["id"]]} for item in INGREDIENTS],
            "recognition": model_status(),
            "session_id": capture_session,
        }

    async def uploaded_image(file):
        try:
            payload = await file.read(MAX_UPLOAD + 1)
        finally:
            await file.close()
        if len(payload) > MAX_UPLOAD:
            raise HTTPException(413, "Please upload an image smaller than 6 MB.")
        try:
            return await asyncio.to_thread(decode_image, payload)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/health")
    async def health():
        return {"ok": True, "model": model_status()}

    @app.get("/api/config")
    async def get_config():
        return config()

    @app.put("/api/config")
    async def update_config(update: ConfigUpdate):
        try:
            await asyncio.to_thread(store.set_durations, update.durations)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return config()

    @app.get("/api/model")
    async def get_model():
        return model_status()

    @app.put("/api/model/provider")
    async def select_provider(update: ProviderUpdate):
        async with inference_lock:
            try:
                next_model = update.model or app.state.openai_model
                next_recognizer = await make_recognizer(update.provider, next_model)
            except Exception:
                message = (
                    "Cloud recognition could not be initialized. Check the cloud model, request limit and requirements-vision.txt."
                    if update.provider == "openai" else
                    "Local recognition could not be initialized. Install the local dependencies and prepare the model."
                )
                raise HTTPException(503, message) from None
            previous_recognizer = app.state.recognizer
            app.state.recognizer = next_recognizer
            app.state.provider = update.provider
            app.state.openai_model = next_model
            app.state.model_error = None
            close = getattr(previous_recognizer, "close", None)
            if close is not None:
                try:
                    await asyncio.to_thread(close)
                except Exception:
                    logger.warning("Recognition client cleanup failed")
        return model_status()

    @app.post("/api/model/reload")
    async def reload_model():
        async with inference_lock:
            try:
                if app.state.recognizer is None:
                    app.state.recognizer = await make_recognizer(app.state.provider)
                else:
                    await asyncio.to_thread(app.state.recognizer.reload)
                app.state.model_error = None
            except Exception as error:
                if app.state.provider == "openai":
                    logger.warning("Cloud recognition reload failed")
                    raise HTTPException(503, "Cloud recognition could not be reloaded. Check the server environment and cloud dependencies.") from None
                logger.exception("Model reload failed")
                raise HTTPException(503, "Model could not be loaded. Check the server log and model files.") from error
        return model_status()

    @app.post("/api/recognize")
    async def recognize(image: UploadFile = File(...)):
        frame = await uploaded_image(image)
        async with inference_lock:
            if model_status()["state"] != "ready":
                raise HTTPException(503, model_status()["message"])
            try:
                return await asyncio.to_thread(app.state.recognizer.predict, frame)
            except Exception as error:
                from .openai_recognition import RecognitionError
                if isinstance(error, RecognitionError):
                    headers = {"Retry-After": str(error.retry_after_seconds)} if error.retry_after_seconds else None
                    raise HTTPException(error.status_code, str(error), headers=headers) from None
                if app.state.provider == "openai":
                    logger.warning("Cloud recognition failed")
                    raise HTTPException(503, "Cloud recognition failed. Retry manually or select the ingredient yourself.") from None
                logger.exception("Recognition failed")
                raise HTTPException(503, "Recognition failed. Try again or reload the model.") from error

    @app.get("/api/dataset")
    async def dataset():
        return await asyncio.to_thread(store.summary)

    @app.post("/api/samples")
    async def save_sample(image: UploadFile = File(...), label: str = Form(...), session_id: str = Form(...)):
        frame = await uploaded_image(image)
        try:
            return await asyncio.to_thread(store.add_sample, frame, label, session_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/dataset/export")
    async def export_dataset():
        path = await asyncio.to_thread(store.export)
        return FileResponse(path, media_type="application/zip", filename="hotpot-dataset.zip", background=BackgroundTask(path.unlink, missing_ok=True))

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "static" / "index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


app = create_app()
