# Hotpot Lab

A prototype with selectable GPT vision and local recognition for **肥牛卷、蟹棒、大白菜、藕片、嫩豆腐、宽粉、乌冬面、香菇、牛肉丸**: camera ingredient recognition, editable countdowns, reminders, labeled-image capture, dataset export, and a MobileNet fine-tuning workflow. The first interaction is **show an ingredient → recognize it → press Added when it enters the pot**. Automatic insertion detection and robotic retrieval are later work.

## Recommended setup: Git + Docker Desktop

Git keeps your code up to date; Docker keeps Python and dependencies inside the container. Stop any previous Python server with **Ctrl+C** in its Terminal to free port 8000. Install [Docker Desktop for Apple Silicon](https://docs.docker.com/desktop/setup/install/mac-install/) and start it. The image supports both Apple Silicon and Intel. The browser accesses your USB camera directly, so the container needs no USB device mapping.

Clone the project once:

```bash
git clone https://github.com/1err/HOTPOT.git
cd HOTPOT
bash scripts/hotpot.sh setup
```

Setup privately prompts for your OpenAI API key in Terminal and stores it in `.secrets/openai_api_key` with permissions limited to your user. You can create or replace it separately:

```bash
python3 scripts/configure_key.py
# To replace an existing saved key:
bash scripts/hotpot.sh key
```

The key is ignored by Git, excluded from Docker builds and ZIPs, and mounted read-only at `/run/secrets/openai_api_key`. It is not included in the container's environment variables. Docker Compose secrets are local file mounts here, not encrypted secret storage; your laptop still holds the original private file. Keep the key on your laptop rather than pasting it into this chat.

Open **http://localhost:8000**. For subsequent code updates:

```bash
bash scripts/hotpot.sh update
```

This runs `git pull --ff-only`, rebuilds the container, and restarts it. It refuses to overwrite local source edits. Camera captures and timer configuration remain in `./data`; browser timer deadlines remain in local storage. `.venv`, models, `.env`, and `.secrets` stay outside Git. Each intentional container restart resets the process's 100-request limit.

Other commands: `start`, `stop`, `status`, and `logs`. `logs` follows output; **Ctrl+C** stops following logs while the container continues running. `stop` preserves data. To change the model, port, or request limit, copy `.env.example` to `.env`, edit those non-secret settings, then run `start`. The helper sets the container's user/group to your local user so private keys and captures stay readable on macOS and Linux.

If migrating from a downloaded prototype, clone to a new folder and copy your previous `data` directory into the clone before setup to keep durations and labeled images. Keep using the same localhost port to retain this browser's timers. The Docker image contains only GPT dependencies; local CLIP/MobileNet remains available through the native Python setup below.

The service binds to `127.0.0.1`, runs without root privileges, has a read-only filesystem with explicit data/tmp mounts, and has a health check. It does not restart automatically and make additional requests. Dependencies and the multiarch Python base image are pinned. CI checks the API/contracts and builds both amd64 and arm64; the amd64 smoke test uses a dummy key without calling OpenAI.

## Native Python setup (optional)

Stop the server with **Ctrl+C**, extract `hotpot-gpt-update.zip`, and merge its `HOTPOT` contents into your **existing project folder**. Keep that folder's `.venv`, `data`, and `models`; an extracted source folder alone does not contain an installed environment. This update excludes those three directories.

From your existing project folder:

```bash
.venv/bin/python -m pip install -r requirements-vision.txt
.venv/bin/python scripts/start_vision.py
```

The launcher privately prompts for an [OpenAI API key](https://platform.openai.com/api-keys). Paste it into Terminal at that prompt; its characters are hidden. It keeps the key in the server process, never in the webpage, source archive, or dataset. If `OPENAI_API_KEY` is already set, it uses that value. Enable API billing, which is [separate from a ChatGPT subscription](https://help.openai.com/en/articles/9039756-managing-billing-for-chatgpt-and-the-api-platform).

Open **http://localhost:8000**, then refresh with **Cmd+Shift+R**. Start the camera, hold one ingredient in the guide, and click **Recognize frame**. GPT mode starts with live recognition switched off. Select **GPT-5.4 mini** in the model chooser to compare the cheaper model, or launch with `--model gpt-5.4-mini`.

For a fresh installation, first run `python3 -m venv .venv`, then the two commands above. GPT-only installation needs no Torch, CLIP weights, or ingredient fine-tuning. Each recognition sends the selected camera-guide crop to OpenAI; the server and timers run on your laptop. Requests use `store: false`, which disables Responses storage but does not override API data-retention policies.

## GPT confidence display

New GPT results include **High / Medium / Low** confidence bands and a short explanation. These are the model's own qualitative assessments, clearly labeled as an uncalibrated GPT estimate. They are not measured probabilities, and the app does not turn them into percentages. Low-confidence results remain uncertain and do not automatically select an ingredient. Older responses without a band keep their existing presentation; local-model match scores are unchanged.

Use labeled evaluations to check whether high-confidence results are actually more reliable. Fine-tuning is not needed to display these bands, and three public photos per ingredient are too few to calibrate them reliably.

## GPT costs and controls

The comparison offers **GPT-5.4** and **GPT-5.4 mini**. These are supported vision models; neither has passed our ingredient tests yet. Start with the larger model to establish a baseline, then compare mini using the same images before choosing it for cost. The browser captures the visible guide crop at up to 768 pixels. GPT gets that crop rather than CLIP's internal 224-pixel input. Its answer is constrained to the nine ingredient IDs or background, with uncertainty and a short reason. GPT results display no invented confidence percentages.

Approximate standard API costs, checked October 9, 2026:

| Model | Input / million tokens | Output / million tokens | Estimate / image | 100 images | 1 hour at one image / 5 seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.4 | $2.50 | $15.00 | $0.003–$0.005 | $0.30–$0.50 | $2.16–$3.60 |
| GPT-5.4 mini | $0.75 | $4.50 | $0.0009–$0.0015 | $0.09–$0.15 | $0.65–$1.08 |

These are planning estimates, not measured bills. A 768-pixel square uses approximately 692 image input tokens under the published patch rules. The estimates also allow roughly 300–600 text/schema input tokens and 40–120 output tokens, without caching discounts. Actual crop sizes and token counts change the bill. See [model prices](https://developers.openai.com/api/docs/models/gpt-5.4), [mini prices](https://developers.openai.com/api/docs/models/gpt-5.4-mini), and [image token rules](https://developers.openai.com/api/docs/guides/images-vision).

The interface estimates cost from returned token usage. Recognition is manual by default; optional live mode waits at least five seconds between checks and pauses on errors. The server enforces a three-second minimum gap and **100 cloud attempts per server process**, shared across model switches and reloads. Restarting resets this limit. It is not a hard dollar budget: interrupted calls can have charges absent from returned usage. Your API usage page is the billing authority. To deliberately allow more attempts, launch with `--request-limit 500`.

## Compare models with your camera images

Use **Save labeled crop** and name the session, for example `first-camera-test`. Include several views of each ingredient and empty/background frames. No fine-tuning is needed. Preview the plan without API calls:

```bash
.venv/bin/python scripts/benchmark_recognition.py --session first-camera-test --models local,gpt-5.4,gpt-5.4-mini --limit 30
```

Add `--run` to execute it. The script reads `OPENAI_API_KEY` or prompts privately for the key. Its report records class coverage, errors, accepted recognition accuracy, abstentions, latency, and estimated token cost. All models receive the same saved images, then apply their own normal preprocessing. Including `local` requires the local dependencies and prepared weights below. Missing classes are reported so a small test is not presented as nine-class accuracy.

The integration is verified using mocked API responses. **GPT accuracy on your ingredients remains unmeasured until the comparison runs with real photos and your API account.**

## Reproduce the 27-photo public-image test

`training_data/public_benchmark.json` fixes three distinct public photograph candidates for each of the nine ingredients. It records each exact source page, image URL, attribution, and review status. These are evaluation inputs, separate from the training-image catalog. The current manifest still needs pixel-level review; no recognition results are implied by its labels.

Preview the complete test without downloading images or calling GPT:

```bash
python scripts/benchmark_public_images.py
```

Download the inputs and open the generated `data/public-benchmark/gallery.html` to inspect the actual images:

```bash
python scripts/benchmark_public_images.py --download-only
```

Verify that each input shows its intended ingredient; an optional normalized `crop_box` records an ingredient-only crop. Mark suitable entries `review_status: "accepted"` before executing. Unsuitable candidates should be replaced with a distinct photograph of the same ingredient. All 27 must be reviewed before paid inference.

```bash
python scripts/benchmark_public_images.py --run --model gpt-5.4
```

The runner uses the same GPT recognizer as the app. It sends image pixels and the common ingredient prompt, without the source filenames, expected labels, or website captions. It records the exact input hashes, expected/predicted ingredient, uncertainty, explanation, model-assessed confidence when available, latency, reported token usage, and estimated cost. It writes JSON, Markdown, and an image gallery under `data/public-benchmark/`. Request errors are reported separately from incorrect predictions. Completed or interrupted attempts are preserved so rerunning the command does not silently repeat paid requests; `--rerun` explicitly archives and repeats the test.

For Docker, run the same script using the app's dependencies and mounted key:

```bash
export HOTPOT_UID="$(id -u)" HOTPOT_GID="$(id -g)"
docker compose run --rm --no-deps \
  --volume "$PWD/scripts:/app/scripts:ro" \
  --volume "$PWD/training_data:/app/training_data:ro" \
  hotpot python scripts/benchmark_public_images.py --download-only
```

After reviewing, replace `--download-only` with `--run --model gpt-5.4`. The normal app image remains small; these source folders are mounted only for this command. The API key stays in its existing secret file. Sources are downloaded locally and never automatically imported into training or committed to Git. One product photograph has no stated open license; its source link is recorded without redistributing its pixels.

Three photos per ingredient can expose obvious confusions but cannot establish reliable per-class accuracy or calibrate confidence. Public photos may also have appeared in pretraining. Follow this with a separate test of your actual webcam images before deciding whether prompting, capture changes, or local model training is needed.

## Start the local recognition model

Python 3.11 or 3.12 is recommended; current dependencies also provide Python 3.13 wheels for Apple Silicon Macs. Run these commands from this directory. The local provider needs no API key or paid inference service.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Linux/Windows CPU installation avoids unnecessary CUDA packages:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/prepare_model.py --mode both
python run.py
```

On macOS, install `torch torchvision` from the normal package index instead of the CPU-index command. On Windows, create the environment with `py -3.12 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1` in PowerShell. The remaining `python` commands are the same.

Open **http://localhost:8000** on the computer running the server. Browser camera access requires localhost or HTTPS. The model downloads once: OpenCLIP is approximately 389 MiB; the MobileNet training initializer is approximately 10 MiB. The server never downloads weights automatically and works offline afterward. An existing prepared workspace can use `.venv/bin/python run.py` directly.

## Try it

1. Enable the camera and choose your USB camera if more than one is connected. An image upload also works.
2. Hold one ingredient inside the guide box. The server receives and recognizes that center crop.
3. Check the observation or select the correct ingredient manually.
4. When you add it to the pot, click **Added to pot**, or press **Space** while no text field is focused.
5. The timer uses your configured duration. Each ingredient permits one active timer. Repeated recognition or repeated start actions do not reset it.
6. When its duration expires, the card stays highlighted until dismissed. Sound and spoken reminders are optional and can be tested in the interface.

Timers are saved in this browser's local storage using absolute deadlines, so refreshes and background tabs do not reset countdowns. They are per browser/device, not synchronized across diners' phones. Alerts from a closed page are delivered when that page is opened again; the app is not an operating-system notification service.

Initial durations supplied by the project owner:

| Ingredient | ID | Duration |
| --- | --- | ---: |
| 肥牛卷 | `beef_rolls` | 15 seconds |
| 蟹棒 | `crab_sticks` | 90 seconds |
| 大白菜 | `napa_cabbage` | 150 seconds |
| 藕片 | `lotus_root` | 300 seconds |
| 嫩豆腐 | `soft_tofu` | 300 seconds |
| 宽粉 | `wide_noodles` | 480 seconds |
| 乌冬面 | `udon_noodles` | 180 seconds |
| 香菇 | `shiitake_mushrooms` | 300 seconds |
| 牛肉丸 | `beef_meatballs` | 480 seconds |

Edit durations in the app; changes affect future timers. The prototype's readiness rule is reaching the selected duration.

## Recognition and improving it

The local baseline is **real pretrained OpenCLIP RN50-quickgelu/YFCC15M**, comparing the image to several descriptions of each ingredient plus background/other-food descriptions. It works without registering plates or training examples first. It is a zero-shot baseline, **not a model trained or evaluated on these nine ingredients**. Relative match scores are not measured recognition accuracy. Ambiguous or background predictions are marked uncertain; manual selection remains available.

After collecting images, you can replace the baseline with a MobileNetV3-Small classifier trained on your labels. The local provider's model-loading preference is:

1. `models/hotpot_mobilenet.pt`: your fine-tuned nine-ingredient + background classifier.
2. `models/clip_rn50_yfcc15m.pt`: the initial zero-shot baseline.
3. ImageNet embedding similarity to labeled examples, if CLIP is unavailable and each label has at least three captured examples.

Capture images in **Build your dataset**. Choose the true label and a recording-session name. Include **background / other** images: empty hands, utensils, plates, and foods outside the nine classes. Changing sessions between days/backgrounds lets the trainer keep whole sessions out of training. Recognition frames are not saved automatically; **Save labeled crop** is explicit. Uploaded/captured examples are re-encoded as JPEG without their original metadata.

Public ingredient images can supplement camera captures. Import a folder layout with the stable IDs:

```text
ingredient-images/
  beef_rolls/*.jpg
  crab_sticks/*.jpg
  napa_cabbage/*.jpg
  lotus_root/*.jpg
  soft_tofu/*.jpg
  wide_noodles/*.jpg
  udon_noodles/*.jpg
  shiitake_mushrooms/*.jpg
  beef_meatballs/*.jpg
  background/*.jpg
```

```bash
python scripts/import_images.py ingredient-images --session public-images-1
python scripts/train.py --epochs 15
```

Use legitimately sourced, correctly labeled images and preserve different capture/source sessions; do not invent separate sessions for adjacent frames to satisfy the split. The trainer requires every label, including background, on both sides of the split and at least eight images per label across at least two sessions. Those are minimum workflow checks, not a recommendation that eight examples are enough for good recognition. Start with varied examples, inspect failures, and collect more.

Training writes an atomic checkpoint and `models/training_report.json` with the session split, confusion matrix, per-class precision/recall, and score calibration. Validation sessions select the best epoch and calibration, so the report is validation performance rather than an independent final accuracy claim. Test a fresh recording session separately. Click **Reload model** after training to use the new checkpoint.

```bash
python scripts/evaluate.py --session kitchen-day-3
```

Useful public-data starting points: [FoodSeg103](https://xiongweiwu.github.io/foodseg103.html) supplies ingredient annotations; [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) focuses on dishes, so its labels do not directly match this vocabulary. This prototype does not bundle either dataset or claim to have trained on them.

## Project layout

```text
hotpot/app.py             FastAPI routes and local static serving
hotpot/recognition.py     CLIP / fine-tuned / example-similarity recognition
hotpot/openai_recognition.py  GPT vision and request/cost accounting
hotpot/training.py        Session-aware MobileNet fine-tuning
hotpot/storage.py         Config and explicitly captured images
static/                  Camera, predictions, timers, capture UI
scripts/                 Weight preparation, image import, training, evaluation
hardware/                Parametric camera mount and printable parts
data/samples.jsonl        Labeled-image manifest (not committed)
models/                  Local prepared/trained checkpoints (not committed)
tests/                   API, data and browser verification
```

The browser accesses its own camera; the Python server processes the frames. The local provider processes frames on the server computer; GPT mode sends the selected crop to OpenAI. Timers and saved examples remain local. If the server is remote, frames first travel to that machine. Choose the provider explicitly in the interface.

`HOTPOT_DATA_DIR` and `HOTPOT_MODEL_DIR` can select alternative local directories. Use one server worker so dataset/config writes stay coordinated. The launcher binds to loopback by default; public hosting and access control are outside this prototype.

CPU inference uses one Torch thread by default. If your laptop benefits from more threads, try `HOTPOT_TORCH_THREADS=4 python run.py` on macOS/Linux, or set the same environment variable in PowerShell. Compare warm inference times shown in the camera view; the first prediction can take longer.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See the browser-check script for end-to-end camera/timer checks and `hardware/README.md` for printing and assembly. Recognition accuracy still needs testing with actual ingredients and your camera; software checks do not establish it.

```bash
# Browser integration uses a temporary dataset and an explicit test model:
python tests/browser_checks.py
# GPT controls use mocked responses and make no paid API calls:
python tests/browser_vision_checks.py
# CHROMIUM_PATH can point to a Chrome/Chromium executable on your machine.
```

The source ZIP includes the application, tests, CAD and STLs. It excludes installed Python packages, captured photos and large model weights; use the preparation command above to download weights when running on a new computer.

## Upgrade an existing installation to nine ingredients

Stop the running app with **Ctrl+C**. Extract `hotpot-update.zip`, then copy its `HOTPOT` folder contents into your existing project folder. This update archive contains source and documentation only: it has no `data/`, `models/`, or `.venv/` directories, so your downloaded weights, captures, custom durations, and installed environment stay in their existing folders.

Run `.venv/bin/python run.py` again on macOS/Linux, or `.\.venv\Scripts\python.exe run.py` on Windows. Refresh the browser; on macOS, **Cmd+Shift+R** forces the updated interface to reload. Previously saved durations remain, and the three new ingredients receive their initial defaults. Existing six-class fine-tuned checkpoints fall back to the pretrained model with an explanation; fine-tune again for the full nine-label vocabulary before using such a checkpoint.

## Public-image starter collection

The curated source list in `training_data/public_sources.json` records candidate photos with their ingredient labels, original pages, authors, licenses, and source groups. It is a small starter collection, not a trained model or a complete training dataset. Download candidates on your own computer, where the image hosts are reachable:

```bash
python scripts/prepare_public_images.py --dry-run
python scripts/prepare_public_images.py
```

See `training_data/README.md` for reviewing candidates and importing approved images. The downloader stages images separately from the active dataset and model. Inspect the pictures before treating the candidate labels as ground truth, add varied examples and background images, and reserve independent sources or recording sessions for evaluation. No source photos or ingredient-finetuned weights are bundled in this update.

## Sources

- [OpenCLIP](https://github.com/mlfoundations/open_clip): pretrained baseline and implementation. The explicit weight-preparation script records source URL, byte count, and SHA256.
- [Torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html): pretrained ImageNet initializer for transfer learning.
