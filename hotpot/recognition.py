"""Local ingredient recognition: real zero-shot CLIP, then a trained MobileNet.

No downloads occur here. A model must have been prepared explicitly. Scores are
model evidence, not calibrated guarantees about the identity of a food.
"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import threading
import time

from PIL import Image, ImageOps

from .training import CLASS_IDS, INGREDIENT_IDS, CHECKPOINT_NAME, eval_transform, make_mobilenet, read_samples


CLIP_CHECKPOINT = "clip_rn50_yfcc15m.pt"
PROMPTS = {
    "beef_rolls": [
        "a photo of raw thinly sliced beef rolls for Chinese hot pot",
        "a photo of curled rolls of red and white marbled beef slices",
        "a photo of thin rolled beef slices held by chopsticks",
    ],
    "crab_sticks": [
        "a photo of red and white imitation crab sticks",
        "a photo of surimi crab sticks for Chinese hot pot",
        "a photo of a red and white crab stick held by chopsticks",
    ],
    "napa_cabbage": [
        "a photo of a raw napa cabbage leaf",
        "a photo of chopped Chinese cabbage with pale green leaves and white stalks",
        "a photo of a piece of napa cabbage held by chopsticks",
    ],
    "lotus_root": [
        "a photo of raw lotus root slices with round holes",
        "a photo of sliced lotus root for Chinese hot pot",
        "a photo of a round lotus root slice held by chopsticks",
    ],
    "soft_tofu": [
        "a photo of a block of soft white tofu",
        "a photo of cubes of silken tofu for Chinese hot pot",
        "a photo of a piece of white tofu held by chopsticks",
    ],
    "wide_noodles": [
        "a photo of wide translucent flat starch noodles for Chinese hot pot",
        "a photo of broad flat sweet potato glass noodles",
        "a photo of wide transparent flat noodles held by chopsticks",
    ],
    "udon_noodles": [
        "a photo of thick round white Japanese udon noodles",
        "a photo of opaque white wheat udon noodles for hot pot",
        "a photo of thick white udon noodles held by chopsticks",
    ],
    "shiitake_mushrooms": [
        "a photo of whole shiitake mushrooms with brown caps and short stems",
        "a photo of a shiitake mushroom with a brown cap and pale gills",
        "a photo of a brown shiitake mushroom held by chopsticks for hot pot",
    ],
    "beef_meatballs": [
        "a photo of round beef meatballs for Chinese hot pot",
        "a photo of smooth light brown Chinese beef balls",
        "a photo of a beef meatball held by chopsticks",
    ],
}
BACKGROUND_PROMPTS = [
    "a photo of an empty plate", "a photo of an empty hand", "a photo of chopsticks without food",
    "a photo of a kitchen tabletop", "a photo of a boiling hot pot of soup", "a photo of a bowl of rice",
    "a photo of raw shrimp", "a photo of white fish balls", "a photo of enoki mushrooms",
    "a photo of white button mushrooms", "a photo of pork meatballs",
    "a photo of potato slices", "a photo of a carrot", "a photo of a bottle",
    "a photo of a computer screen", "a photo of a person's face", "a photo of a white wall",
]


def validate_ingredient_checkpoint(checkpoint: dict) -> None:
    """Reject an outdated classifier rather than creating untrained extra outputs."""
    if checkpoint.get("arch") != "mobilenet_v3_small" or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint architecture or format")
    saved_labels = tuple(checkpoint.get("class_ids", []))
    if saved_labels != CLASS_IDS:
        missing = [label for label in CLASS_IDS if label not in saved_labels]
        extra = [label for label in saved_labels if label not in CLASS_IDS]
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("obsolete " + ", ".join(extra))
        if not details:
            details.append("label order differs")
        raise ValueError(
            "Saved ingredient checkpoint does not match the current ingredient vocabulary ("
            + "; ".join(details) + "). Retrain it with all current ingredients and background."
        )


class Recognizer:
    def __init__(self, data_dir: Path, model_dir: Path):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self._lock = threading.RLock()
        self._model = None
        self._transform = None
        self._mode = "unavailable"
        self._text_features = None
        self._prototypes = None
        self._metadata = {}
        self._message = "No prepared recognition model."
        self._warnings = []
        self._state = "unavailable"
        self.reload()

    def status(self) -> dict:
        with self._lock:
            labels = {"clip": "OpenCLIP RN50 · zero-shot baseline",
                      "finetuned": "MobileNetV3 · ingredient fine-tune",
                      "fewshot": "MobileNetV3 · example similarity",
                      "unavailable": "Model not prepared"}
            return {
                "state": self._state, "ready": self._state == "ready", "mode": self._mode,
                "label": labels[self._mode], "message": self._message,
                "warnings": list(self._warnings), "metadata": self._metadata,
                "class_ids": list(CLASS_IDS),
            }

    def reload(self) -> dict:
        with self._lock:
            self._state = "loading"
            self._mode = "unavailable"
            self._model = None
            self._transform = None
            self._metadata = {}
            self._warnings = []
            candidates = [
                (self.model_dir / CHECKPOINT_NAME, self._load_finetuned),
                (self.model_dir / CLIP_CHECKPOINT, self._load_clip),
            ]
            checkpoint_failure = None
            for path, loader in candidates:
                if not path.exists():
                    continue
                try:
                    loader(path)
                    if self._mode == "clip" and checkpoint_failure:
                        self._metadata["fallback_reason"] = checkpoint_failure
                        self._message = (
                            "Using the pretrained zero-shot baseline because the saved ingredient "
                            "checkpoint could not be loaded. " + checkpoint_failure
                        )
                    self._state = "ready"
                    return self.status()
                except Exception as error:
                    self._warnings.append(f"Could not load {path.name}: {type(error).__name__}: {error}")
                    if path.name == CHECKPOINT_NAME:
                        checkpoint_failure = str(error)
            initializer = self.model_dir / "mobilenet_v3_small_imagenet.pth"
            if initializer.exists() and (self.data_dir / "samples.jsonl").exists():
                try:
                    self._load_fewshot(initializer)
                    self._state = "ready"
                    return self.status()
                except Exception as error:
                    self._warnings.append(f"Example similarity unavailable: {error}")
            self._model = None
            self._mode = "unavailable"
            self._state = "unavailable"
            self._message = "Run python scripts/prepare_model.py --mode clip, then reload the model."
            return self.status()

    def _runtime(self):
        import torch
        # One thread leaves headroom for camera/API work and was consistently
        # responsive in our CPU smoke benchmark. Larger laptops can opt into
        # more threads; excessive parallelism can hurt a batch-of-one workload.
        try:
            threads = int(os.environ.get("HOTPOT_TORCH_THREADS", "1"))
        except ValueError as error:
            raise ValueError("HOTPOT_TORCH_THREADS must be an integer between 1 and 32") from error
        if not 1 <= threads <= 32:
            raise ValueError("HOTPOT_TORCH_THREADS must be an integer between 1 and 32")
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        return torch

    def _load_finetuned(self, path: Path):
        torch = self._runtime()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        validate_ingredient_checkpoint(checkpoint)
        model = make_mobilenet()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        self._model = model
        self._transform = eval_transform()
        self._metadata = checkpoint.get("metadata", {})
        self._mode = "finetuned"
        self._message = "Trained on your labeled examples. Low-confidence or background results are rejected."

    def _load_clip(self, path: Path):
        missing_prompts = [label for label in INGREDIENT_IDS if label not in PROMPTS]
        if missing_prompts:
            raise ValueError("Missing recognition prompts for: " + ", ".join(missing_prompts))
        torch = self._runtime()
        import open_clip
        from open_clip.model import CLIP
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        # Construct the architecture and explicitly load our local, safely decoded
        # checkpoint. This also avoids the factory's misleading "random model"
        # warning before a checkpoint is loaded separately.
        model = CLIP(**open_clip.get_model_config("RN50"), quick_gelu=True)
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
            ),
        ])
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        for obsolete_key in ("input_resolution", "context_length", "vocab_size"):
            state_dict.pop(obsolete_key, None)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        tokenizer = open_clip.get_tokenizer("RN50")
        features = []
        with torch.inference_mode():
            for label in INGREDIENT_IDS:
                embeddings = model.encode_text(tokenizer(PROMPTS[label]))
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                average = embeddings.mean(dim=0)
                features.append(average / average.norm())
            background = model.encode_text(tokenizer(BACKGROUND_PROMPTS))
            background = background / background.norm(dim=-1, keepdim=True)
        self._model, self._transform = model, transform
        self._text_features = torch.stack(features)
        self._background_features = background
        self._mode = "clip"
        self._metadata = {
            "architecture": "RN50-quickgelu", "pretraining": "YFCC15M", "ingredient_training": False,
            "ingredient_class_ids": list(INGREDIENT_IDS), "prompt_version": 2,
            "score_note": "Relative text-image similarity; not an empirically measured probability.",
            "confidence_threshold": 0.45, "similarity_threshold": 0.22, "similarity_margin_threshold": 0.015,
        }
        self._message = "Real pretrained zero-shot baseline. It has not been fine-tuned or evaluated on your ingredients."

    def _load_fewshot(self, path: Path):
        torch = self._runtime()
        from torchvision.models import mobilenet_v3_small
        samples = read_samples(self.data_dir)
        counts = Counter(s["label"] for s in samples)
        if any(counts[label] < 3 for label in CLASS_IDS):
            raise ValueError("Need at least 3 labeled examples for each ingredient and background")
        model = mobilenet_v3_small(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.classifier = torch.nn.Identity()
        model.eval()
        transform = eval_transform()
        prototypes = []
        with torch.inference_mode():
            for label in CLASS_IDS:
                # Limit initial reload work; collecting more examples should lead to
                # actual fine-tuning, not an ever-growing live inference index.
                selected = [s for s in samples if s["label"] == label][-24:]
                vectors = []
                for sample in selected:
                    with Image.open(sample["absolute_path"]) as image:
                        tensor = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
                    embedding = model(tensor).squeeze(0)
                    vectors.append(embedding / embedding.norm())
                prototype = torch.stack(vectors).mean(dim=0)
                prototypes.append(prototype / prototype.norm())
        self._model, self._transform = model, transform
        self._prototypes = torch.stack(prototypes)
        self._mode = "fewshot"
        self._metadata = {"example_counts": dict(counts), "ingredient_training": False,
                          "score_note": "Similarity to captured examples; thresholds are provisional."}
        self._message = "ImageNet embedding similarity to your examples. Fine-tune for an evaluated ingredient model."

    def predict(self, image: Image.Image) -> dict:
        with self._lock:
            if self._state != "ready" or self._model is None:
                return {"ingredient_id": None, "scores": [], "confidence": 0.0, "margin": 0.0,
                        "uncertain": True, "mode": "unavailable", "inference_ms": 0.0,
                        "message": self._message}
            torch = self._runtime()
            started = time.perf_counter()
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self._transform(image).unsqueeze(0)
            raw_margin = None
            similarity = None
            with torch.inference_mode():
                if self._mode == "clip":
                    embedding = self._model.encode_image(tensor)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    known = (embedding @ self._text_features.T).squeeze(0)
                    background = (embedding @ self._background_features.T).max()
                    similarities = torch.cat([known, background.reshape(1)])
                    # Fixed temperature makes relative evidence legible without
                    # implying that the pretrained model's logit scale is calibrated.
                    probabilities = (similarities * 30.0).softmax(dim=0)
                    raw_top = similarities.topk(2).values
                    raw_margin = float(raw_top[0] - raw_top[1])
                    similarity = float(raw_top[0])
                elif self._mode == "finetuned":
                    temperature = self._metadata.get("calibration", {}).get("temperature", 1.0)
                    probabilities = (self._model(tensor).squeeze(0) / temperature).softmax(dim=0)
                else:
                    embedding = self._model(tensor)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    similarities = (embedding @ self._prototypes.T).squeeze(0)
                    probabilities = (similarities * 20.0).softmax(dim=0)
                    raw_top = similarities.topk(2).values
                    raw_margin, similarity = float(raw_top[0] - raw_top[1]), float(raw_top[0])
            top = probabilities.topk(2)
            winner_index = int(top.indices[0])
            winner = CLASS_IDS[winner_index]
            confidence = float(top.values[0])
            margin = float(top.values[0] - top.values[1])
            if self._mode == "clip":
                uncertain = confidence < 0.45 or similarity < 0.22 or raw_margin < 0.015
            elif self._mode == "finetuned":
                calibration = self._metadata.get("calibration", {})
                threshold = calibration.get("confidence_thresholds", {}).get(winner, 0.85)
                uncertain = confidence < threshold or margin < calibration.get("margin_threshold", 0.10)
            else:
                uncertain = confidence < 0.55 or similarity < 0.65 or raw_margin < 0.04
            uncertain = uncertain or winner == "background"
            scores = sorted([
                {"id": label, "score": round(float(probabilities[index]), 5)}
                for index, label in enumerate(CLASS_IDS)
            ], key=lambda item: item["score"], reverse=True)
            result = {
                "ingredient_id": winner if winner != "background" else None,
                "scores": scores, "confidence": round(confidence, 5), "margin": round(margin, 5),
                "uncertain": bool(uncertain), "mode": self._mode,
                "inference_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            if similarity is not None:
                result["similarity"] = round(similarity, 5)
                result["similarity_margin"] = round(raw_margin, 5)
            return result
