"""A fair comparison must report failures and missing classes without API calls."""

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts import benchmark_recognition as benchmark


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    monkeypatch.setattr("hotpot.credentials.DEFAULT_KEY_FILE", tmp_path / "missing-key")


def dataset(tmp_path, labels, *, session="fresh"):
    rows = []
    tmp_path.mkdir(parents=True, exist_ok=True)
    for index, label in enumerate(labels):
        filename = f"sample-{index:03d}.jpg"
        Image.new("RGB", (32, 24), (index * 23 % 256, 75, 120)).save(tmp_path / filename)
        rows.append({"sample_id": f"id-{index}", "path": filename,
                     "label": label, "session_id": session})
    (tmp_path / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return benchmark.select_samples(tmp_path, [session], 100)


def prediction(label, *, uncertain=False):
    return {"predicted_id": label, "ingredient_id": None if label == "background" else label,
            "scores": [], "uncertain": uncertain, "confidence": None, "model": "fake"}


class FakeRecognizer:
    def __init__(self, results=(), *, status=None, ledger=None, cost=0):
        self.results = list(results)
        self.index = 0
        self.seen = []
        self.status_value = status or {"ready": True, "mode": "clip", "metadata": {}}
        self.ledger = ledger
        self.cost = cost

    def status(self):
        result = dict(self.status_value)
        if self.ledger is not None:
            result["usage"] = dict(self.ledger)
        return result

    def predict(self, image):
        self.seen.append(image.copy())
        if self.ledger is not None:
            self.ledger["attempts"] += 1
            self.ledger["completed"] += 1
            self.ledger["input_tokens"] += 100
            self.ledger["output_tokens"] += 10
            self.ledger["estimated_usd"] += self.cost
        result = self.results[self.index]
        self.index += 1
        image.paste((255, 0, 0), (0, 0, image.width, image.height))
        if isinstance(result, Exception):
            raise result
        return result


def test_default_dry_plan_never_builds_models_or_needs_an_api_key(tmp_path, monkeypatch, capsys):
    dataset(tmp_path, ["beef_rolls", "background"])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(benchmark, "build_recognizers", lambda *args: pytest.fail("Dry run made a model"))
    assert benchmark.main(["--data", str(tmp_path), "--session", "fresh",
                           "--models", "local,gpt-5.4,gpt-5.4-mini"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "plan"
    assert report["planned_api_calls"] == 4
    assert report["api_calls_per_model"] == {"local": 0, "gpt-5.4": 2, "gpt-5.4-mini": 2}
    assert report["all_nine_ingredients_present"] is False
    assert not (tmp_path / "benchmark_report.json").exists()


def test_limit_balances_classes_and_identifies_untested_labels(tmp_path):
    dataset(tmp_path, ["beef_rolls"] * 6 + ["crab_sticks"] * 4 + ["background"] * 2)
    selected = benchmark.select_samples(tmp_path, ["fresh"], 5)
    assert [row["label"] for row in selected] == [
        "beef_rolls", "crab_sticks", "background", "beef_rolls", "crab_sticks",
    ]
    plan = benchmark.make_plan(selected, ["local"], ["fresh"])
    assert plan["class_counts"]["napa_cabbage"] == 0
    assert "napa_cabbage" in plan["missing_classes"]
    with pytest.raises(ValueError, match="No samples"):
        benchmark.select_samples(tmp_path, ["absent"], 30)


def test_every_model_receives_identical_saved_pixels_despite_a_mutating_predictor(tmp_path):
    samples = dataset(tmp_path, ["beef_rolls", "background"])
    local = FakeRecognizer([prediction("beef_rolls"), prediction("background", uncertain=True)])
    cloud = FakeRecognizer([prediction("beef_rolls"), prediction("background", uncertain=True)])
    report = benchmark.run_benchmark(samples, {"local": local, "gpt-5.4": cloud}, ["fresh"],
                                     interval_seconds=0)
    for local_image, cloud_image in zip(local.seen, cloud.seen):
        assert local_image.size == cloud_image.size
        assert local_image.tobytes() == cloud_image.tobytes()
    for model in ("local", "gpt-5.4"):
        result = report["results"][model]
        assert result["metrics"]["top1_accuracy"] == 1
        assert result["metrics"]["accepted_accuracy"] == 1
        assert result["metrics"]["accepted_coverage"] == 0.5
        assert result["metrics"]["background_rejection_rate"] == 1
        for sample, row in zip(samples, result["predictions"]):
            assert row["source_sha256"] == hashlib.sha256(Path(sample["absolute_path"]).read_bytes()).hexdigest()
    encoded = json.dumps(report)
    assert "_image_bytes" not in encoded and "base64" not in encoded


def test_api_errors_are_separate_from_accuracy_and_arbitrary_exception_text_is_not_saved(tmp_path):
    samples = dataset(tmp_path, ["beef_rolls", "beef_rolls", "beef_rolls"])
    cloud = FakeRecognizer([prediction("beef_rolls"), RuntimeError("sk-secret https://evil.invalid/imagebase64"),
                            prediction("crab_sticks")])
    result = benchmark.run_benchmark(samples, {"gpt-5.4": cloud}, ["fresh"], interval_seconds=0)["results"]["gpt-5.4"]
    metrics = result["metrics"]
    assert metrics["successful_requests"] == 2 and metrics["errors"] == 1
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["accepted_accuracy"] == 0.5
    assert metrics["per_class"]["beef_rolls"]["errors"] == 1
    assert metrics["all_nine_ingredients_evaluated"] is False
    assert "sk-secret" not in json.dumps(result)
    assert "evil.invalid" not in json.dumps(result)


def test_uncertain_null_does_not_count_as_correct_background(tmp_path):
    samples = dataset(tmp_path, ["background"])
    model = FakeRecognizer([{"ingredient_id": None, "scores": [], "uncertain": True}])
    metrics = benchmark.run_benchmark(samples, {"local": model}, ["fresh"])["results"]["local"]["metrics"]
    assert metrics["classified"] == 0 and metrics["unclassified"] == 1
    assert metrics["top1_accuracy"] == 0
    assert metrics["accepted_accuracy"] is None
    background_index = benchmark.CLASS_IDS.index("background")
    assert metrics["confusion_matrix"][background_index][-1] == 1


def test_rejected_local_top_score_is_retained_for_top1_accuracy(tmp_path):
    samples = dataset(tmp_path, ["beef_rolls"])
    model = FakeRecognizer([{"ingredient_id": "beef_rolls", "scores": [{"id": "beef_rolls", "score": 0.4}],
                             "uncertain": True, "confidence": 0.4}])
    metrics = benchmark.run_benchmark(samples, {"local": model}, ["fresh"])["results"]["local"]["metrics"]
    assert metrics["top1_accuracy"] == 1
    assert metrics["accepted_coverage"] == 0
    assert metrics["accepted_accuracy"] is None


def test_unreadable_last_image_stops_before_any_prediction(tmp_path):
    samples = dataset(tmp_path, ["beef_rolls", "background"])
    Path(samples[-1]["absolute_path"]).write_bytes(b"not an image")
    model = FakeRecognizer([prediction("beef_rolls")])
    with pytest.raises(ValueError, match="Unreadable benchmark image"):
        benchmark.run_benchmark(samples, {"gpt-5.4": model}, ["fresh"])
    assert model.index == 0


def test_unready_second_model_stops_before_the_first_models_paid_prediction(tmp_path):
    samples = dataset(tmp_path, ["beef_rolls"])
    ready = FakeRecognizer([prediction("beef_rolls")])
    unavailable = FakeRecognizer(status={"ready": False})
    with pytest.raises(ValueError, match="not ready"):
        benchmark.run_benchmark(samples, {"gpt-5.4": ready, "local": unavailable}, ["fresh"])
    assert ready.index == 0


@pytest.mark.parametrize("status", [
    {"ready": True, "mode": "fewshot"},
    {"ready": True, "mode": "finetuned", "metadata": {"train_sessions": ["fresh"]}},
    {"ready": True, "mode": "finetuned", "metadata": {"validation_sessions": ["fresh"]}},
])
def test_local_reference_or_training_leakage_is_rejected_before_cloud_calls(tmp_path, status):
    samples = dataset(tmp_path, ["beef_rolls"])
    cloud = FakeRecognizer([prediction("beef_rolls")])
    with pytest.raises(ValueError):
        benchmark.run_benchmark(samples, {"gpt-5.4": cloud, "local": FakeRecognizer(status=status)}, ["fresh"])
    assert cloud.index == 0


def test_missing_key_and_missing_sdk_fail_preflight_without_constructing_local_model(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        benchmark.build_recognizers(["local", "gpt-5.4"], tmp_path, tmp_path, ["fresh"], 2)
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-test-only")
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(ValueError, match="SDK is missing"):
        benchmark.build_recognizers(["local", "gpt-5.4"], tmp_path, tmp_path, ["fresh"], 2)


def test_cloud_spacing_is_separate_from_measured_prediction_latency(tmp_path):
    samples = dataset(tmp_path, ["beef_rolls", "background"])
    model = FakeRecognizer([prediction("beef_rolls"), prediction("background", uncertain=True)])
    now = [10.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    result = benchmark.run_benchmark(samples, {"gpt-5.4": model}, ["fresh"],
                                     clock=lambda: now[0], sleep=sleep)["results"]["gpt-5.4"]
    assert waits == [3.0]
    assert result["metrics"]["mean_latency_ms"] == 0


def test_shared_usage_ledger_reports_per_model_cost_not_cumulative_cost(tmp_path):
    samples = dataset(tmp_path, ["beef_rolls"])
    ledger = {"attempts": 0, "completed": 0, "input_tokens": 0, "output_tokens": 0,
              "estimated_usd": 0.0, "unreported_attempts": 0}
    standard = FakeRecognizer([prediction("beef_rolls")], ledger=ledger, cost=0.002)
    mini = FakeRecognizer([prediction("beef_rolls")], ledger=ledger, cost=0.001)
    results = benchmark.run_benchmark(samples, {"gpt-5.4": standard, "gpt-5.4-mini": mini}, ["fresh"],
                                      interval_seconds=0)["results"]
    assert results["gpt-5.4"]["usage"]["estimated_usd"] == pytest.approx(0.002)
    assert results["gpt-5.4-mini"]["usage"]["estimated_usd"] == pytest.approx(0.001)
    assert results["gpt-5.4-mini"]["usage"]["attempts"] == 1
    assert results["gpt-5.4-mini"]["usage"]["input_tokens"] == 100


def test_invalid_model_and_conflicting_execution_flags_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="distinct models"):
        benchmark.parse_models("local,local")
    with pytest.raises(ValueError, match="Unsupported models"):
        benchmark.parse_models("unknown-model")
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--data", str(tmp_path), "--session", "fresh", "--run", "--dry-run"])
    assert error.value.code == 2


def test_hidden_key_prompt_is_only_used_in_interactive_explicit_execution(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(benchmark.sys.stdin, "isatty", lambda: True)
    prompts = []

    def private_prompt(prompt):
        prompts.append(prompt)
        return "private-key-for-test-only"

    monkeypatch.setattr(benchmark.getpass, "getpass", private_prompt)
    benchmark.ensure_cloud_credentials(["local"])
    assert not prompts
    benchmark.ensure_cloud_credentials(["gpt-5.4"])
    assert len(prompts) == 1 and "hidden" in prompts[0]
    benchmark.ensure_cloud_credentials(["gpt-5.4"])
    assert len(prompts) == 1
    assert "private-key-for-test-only" not in capsys.readouterr().out


def test_secret_file_preflight_needs_no_prompt_or_key_environment(tmp_path, monkeypatch):
    import os

    private_file = tmp_path / "private-key"
    private_file.write_text("project-dummy-secret\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(private_file))
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(benchmark.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(benchmark.getpass, "getpass", lambda *_: pytest.fail("Unexpected key prompt"))
    benchmark.ensure_cloud_credentials(["gpt-5.4"])
    assert "OPENAI_API_KEY" not in os.environ


def test_missing_sdk_or_noninteractive_input_does_not_prompt_or_echo_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(benchmark.getpass, "getpass", lambda *_: pytest.fail("Unexpected key prompt"))
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(ValueError, match="SDK is missing"):
        benchmark.ensure_cloud_credentials(["gpt-5.4"])
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(benchmark.sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError, match="interactive Terminal"):
        benchmark.ensure_cloud_credentials(["gpt-5.4"])


def test_blank_key_prompt_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(benchmark.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(benchmark.getpass, "getpass", lambda *_: "   ")
    with pytest.raises(ValueError, match="no recognition requests"):
        benchmark.ensure_cloud_credentials(["gpt-5.4"])


def test_failed_api_requests_do_not_claim_all_nine_classes_were_evaluated(tmp_path):
    samples = dataset(tmp_path, list(benchmark.CLASS_IDS))
    results = [prediction(label, uncertain=label == "background") for label in benchmark.CLASS_IDS]
    results[0] = RuntimeError("failed")
    model = FakeRecognizer(results)
    report = benchmark.run_benchmark(samples, {"gpt-5.4": model}, ["fresh"], interval_seconds=0)
    assert report["all_nine_ingredients_present"] is True
    metrics = report["results"]["gpt-5.4"]["metrics"]
    assert metrics["all_nine_ingredients_evaluated"] is False
    assert metrics["unevaluated_classes"] == ["beef_rolls"]
