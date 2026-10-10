"""Session-aware transfer learning for the shared ingredient vocabulary and background."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

from .ingredients import INGREDIENT_IDS, SAMPLE_LABELS

CLASS_IDS = tuple(SAMPLE_LABELS)
CHECKPOINT_NAME = "hotpot_mobilenet.pt"


def read_samples(data_dir: Path) -> list[dict]:
    """Read captured/imported images without permitting paths outside the data directory."""
    data_dir = Path(data_dir).resolve()
    manifest = data_dir / "samples.jsonl"
    if not manifest.exists():
        raise ValueError(f"No dataset yet: {manifest}. Capture or import labeled images first.")
    samples = []
    seen = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record["label"] not in CLASS_IDS:
                raise ValueError(f"Unknown label {record['label']!r}; use one of {CLASS_IDS}")
            if not record.get("session_id"):
                raise ValueError("Missing session_id; session IDs are required to prevent frame leakage")
            path = (data_dir / record["path"]).resolve()
            if not path.is_relative_to(data_dir) or not path.is_file():
                raise ValueError(f"Missing image or path outside data directory: {record['path']}")
            if path in seen:
                raise ValueError(f"Duplicate image path: {record['path']}")
            seen.add(path)
            samples.append({**record, "absolute_path": str(path)})
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Invalid sample manifest line {line_number}: {error}") from error
    return samples


def split_by_session(
    samples: list[dict], *, validation_fraction: float = 0.25, seed: int = 42,
    minimum_images_per_class: int = 8,
) -> tuple[list[dict], list[dict]]:
    """Keep complete collection sessions together and require all labels on both sides."""
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5")
    counts = Counter(sample["label"] for sample in samples)
    problems = []
    for label in CLASS_IDS:
        sessions = {sample["session_id"] for sample in samples if sample["label"] == label}
        if counts[label] < minimum_images_per_class:
            problems.append(f"{label}: {counts[label]} images; need at least {minimum_images_per_class}")
        if len(sessions) < 2:
            problems.append(f"{label}: {len(sessions)} sessions; need at least 2 separate collection sessions")
    if problems:
        raise ValueError("Dataset is not ready for a meaningful split:\n" + "\n".join(problems))

    sessions = sorted({sample["session_id"] for sample in samples})
    rng = random.Random(seed)
    best: tuple[float, list[dict], list[dict]] | None = None
    # Randomized grouped search accommodates both a session with all classes and
    # independently imported per-class sessions. A whole session always stays together.
    for attempt in range(1500):
        if attempt % 3 == 0:
            k = max(1, min(len(sessions) - 1, round(len(sessions) * validation_fraction)))
            validation_sessions = set(rng.sample(sessions, k))
        else:
            validation_sessions = {session for session in sessions if rng.random() < validation_fraction}
        train = [s for s in samples if s["session_id"] not in validation_sessions]
        validation = [s for s in samples if s["session_id"] in validation_sessions]
        train_counts = Counter(s["label"] for s in train)
        validation_counts = Counter(s["label"] for s in validation)
        if any(train_counts[c] < 3 or validation_counts[c] < 2 for c in CLASS_IDS):
            continue
        error = sum(abs(validation_counts[c] / counts[c] - validation_fraction) for c in CLASS_IDS)
        if best is None or error < best[0]:
            best = (error, train, validation)
    if best is None:
        raise ValueError(
            "Cannot keep whole sessions separate while retaining every class in training and validation. "
            f"Collect another independent session containing all {len(CLASS_IDS)} labels, including background."
        )
    return best[1], best[2]


def make_mobilenet(*, checkpoint: Path | None = None, class_count: int | None = None):
    """Construct locally only. ImageNet initialization is explicit and never downloaded here."""
    import torch
    from torch import nn
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=None)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if class_count is None:
        class_count = len(CLASS_IDS)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    return model


def eval_transform():
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _metrics(logits, targets, temperature: float) -> dict:
    import torch
    probabilities = (logits / temperature).softmax(dim=1)
    predicted = probabilities.argmax(dim=1)
    matrix = torch.zeros(len(CLASS_IDS), len(CLASS_IDS), dtype=torch.int64)
    for actual, guess in zip(targets.tolist(), predicted.tolist()):
        matrix[actual, guess] += 1
    per_class = {}
    recalls = []
    for index, label in enumerate(CLASS_IDS):
        correct = int(matrix[index, index])
        support = int(matrix[index].sum())
        guessed = int(matrix[:, index].sum())
        recall = correct / support if support else 0.0
        recalls.append(recall)
        per_class[label] = {
            "support": support, "precision": correct / guessed if guessed else 0.0, "recall": recall,
        }
    confidence = probabilities.max(dim=1).values
    correct = predicted.eq(targets).float()
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        mask = (confidence > lower) & (confidence <= lower + 0.1)
        if mask.any():
            ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs())
    return {
        "class_ids": list(CLASS_IDS), "confusion_matrix": matrix.tolist(),
        "accuracy": float(correct.mean()), "macro_recall": sum(recalls) / len(recalls),
        "per_class": per_class, "expected_calibration_error": ece,
        "nll": float(torch.nn.functional.cross_entropy(logits / temperature, targets)),
    }


def _calibrate(logits, targets) -> dict:
    import torch
    # Small validation sets don't justify a large calibration model. Fit one scalar
    # temperature and retain explicit conservative rejection thresholds.
    candidates = torch.linspace(0.5, 3.0, 51).tolist()
    temperature = min(candidates, key=lambda t: float(torch.nn.functional.cross_entropy(logits / t, targets)))
    probabilities = (logits / temperature).softmax(dim=1)
    confidence, predicted = probabilities.max(dim=1)
    top2 = probabilities.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    thresholds = {}
    for index, label in enumerate(CLASS_IDS):
        threshold = 0.85
        selected = predicted.eq(index) & (margin >= 0.10)
        options = sorted(set(confidence[selected].tolist()))
        for candidate in options:
            accepted = selected & (confidence >= max(candidate, 0.50))
            if int(accepted.sum()) >= 3 and float(predicted[accepted].eq(targets[accepted]).float().mean()) >= 0.90:
                threshold = max(candidate, 0.50)
                break
        thresholds[label] = round(threshold, 4)
    return {
        "temperature": temperature, "confidence_thresholds": thresholds,
        "margin_threshold": 0.10,
        "note": "Thresholds tuned on session-held-out validation, not an independent test set. "
                "Confidence remains a model score; new foods can still be confidently misclassified.",
    }


def train_model(
    data_dir: Path, model_dir: Path, *, epochs: int = 15, batch_size: int = 16,
    learning_rate: float = 0.001, unfreeze_last: int = 2, validation_fraction: float = 0.25,
    seed: int = 42, minimum_images_per_class: int = 8,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image, ImageOps

    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs and batch_size must be positive, and learning_rate must be greater than zero")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    samples = read_samples(data_dir)
    train, validation = split_by_session(
        samples, validation_fraction=validation_fraction, seed=seed,
        minimum_images_per_class=minimum_images_per_class,
    )
    initializer = model_dir / "mobilenet_v3_small_imagenet.pth"
    if not initializer.exists():
        raise ValueError("ImageNet initializer missing. Run: python scripts/prepare_model.py --mode mobilenet")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class IngredientDataset(Dataset):
        def __init__(self, records, transform):
            self.records, self.transform = records, transform
        def __len__(self):
            return len(self.records)
        def __getitem__(self, index):
            record = self.records[index]
            with Image.open(record["absolute_path"]) as image:
                tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
            return tensor, CLASS_IDS.index(record["label"])

    # Fail before training if an imported image is unreadable.
    for sample in samples:
        try:
            with Image.open(sample["absolute_path"]) as image:
                image.verify()
        except Exception as error:
            raise ValueError(f"Unreadable training image: {sample['path']}: {error}") from error

    train_loader = DataLoader(IngredientDataset(train, train_transform), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(IngredientDataset(validation, eval_transform()), batch_size=batch_size, num_workers=0)
    model = make_mobilenet(checkpoint=initializer)
    if not 0 <= unfreeze_last <= len(model.features):
        raise ValueError(f"unfreeze_last must be between 0 and {len(model.features)}")
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if unfreeze_last:
        for block in model.features[-unfreeze_last:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    train_counts = Counter(s["label"] for s in train)
    weights = torch.tensor([len(train) / (len(CLASS_IDS) * train_counts[c]) for c in CLASS_IDS])
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.0001)
    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        # Keep running statistics in frozen feature blocks frozen too.
        frozen_end = len(model.features) - unfreeze_last
        for block in model.features[:frozen_end]:
            block.eval()
        losses = []
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        all_logits, all_targets = [], []
        with torch.inference_mode():
            for images, targets in validation_loader:
                all_logits.append(model(images))
                all_targets.append(targets)
        logits, targets = torch.cat(all_logits), torch.cat(all_targets)
        validation_loss = float(torch.nn.functional.cross_entropy(logits, targets))
        accuracy = float(logits.argmax(dim=1).eq(targets).float().mean())
        record = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses),
                  "validation_loss": validation_loss, "validation_accuracy": accuracy}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    all_logits, all_targets = [], []
    with torch.inference_mode():
        for images, targets in validation_loader:
            all_logits.append(model(images))
            all_targets.append(targets)
    logits, targets = torch.cat(all_logits), torch.cat(all_targets)
    calibration = _calibrate(logits, targets)
    report = _metrics(logits, targets, calibration["temperature"])
    report.update({
        "calibration": calibration, "history": history,
        "train_samples": len(train), "validation_samples": len(validation),
        "train_sessions": sorted({s["session_id"] for s in train}),
        "validation_sessions": sorted({s["session_id"] for s in validation}),
        "evaluation_note": "All validation sessions were excluded from gradient training. "
            "They were used to choose the best epoch and calibrate scores; evaluate a fresh meal separately.",
    })
    fingerprint = hashlib.sha256(json.dumps([
        {k: s[k] for k in ("path", "label", "session_id")} for s in samples
    ], sort_keys=True).encode()).hexdigest()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_fingerprint": fingerprint,
        "train_samples": len(train), "validation_samples": len(validation),
        "validation_accuracy": report["accuracy"], "macro_recall": report["macro_recall"],
        "calibration": calibration, "seed": seed,
        "train_sessions": report["train_sessions"], "validation_sessions": report["validation_sessions"],
    }
    checkpoint = {"format_version": 1, "arch": "mobilenet_v3_small", "class_ids": list(CLASS_IDS),
                  "state_dict": best_state, "metadata": metadata}
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / CHECKPOINT_NAME
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)
    report_path = model_dir / "training_report.json"
    report_tmp = report_path.with_suffix(".json.tmp")
    report_tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(report_tmp, report_path)
    return {"checkpoint": str(destination), "report_path": str(report_path), **report}
