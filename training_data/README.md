# Public ingredient image starter

`public_sources.json` records **35 photograph candidates** with source page, direct image URL, author, license, ingredient label, photographer group, and composition notes. The source pages identify the ingredients and specify the licenses. **The pixels have not yet been reviewed, downloaded, or used to train a model in this workspace.**

| Label | Ingredient | Candidates | Photographer/source groups |
| --- | --- | ---: | ---: |
| `beef_rolls` | 肥牛卷 | 2 | 1 |
| `crab_sticks` | 蟹棒 | 6 | 6 |
| `napa_cabbage` | 大白菜 | 4 | 4 |
| `lotus_root` | 藕片 | 1 | 1 |
| `soft_tofu` | 嫩豆腐 | 2 | 2 |
| `wide_noodles` | 宽粉 | 2 | 2 |
| `udon_noodles` | 乌冬面 | 2 | 2 |
| `shiitake_mushrooms` | 香菇 | 8 | 8 |
| `beef_meatballs` | 牛肉丸 | 4 | 4 |
| `background` | Empty plate, chopsticks, apple, tomatoes | 4 | 4 |

This is a starting collection for review and supplementing camera captures. It is too small to establish good recognition: several classes have one or two photos, and both beef-roll photos share a photographer. Splitting or augmenting one photo does not create independent evidence. The current zero-shot recognizer remains available while a stronger dataset is collected.

## Download on your Mac

From the prototype folder after installing its Python dependencies:

```bash
# Offline validation and counts; performs no downloads and creates no files.
.venv/bin/python scripts/prepare_public_images.py --dry-run

# Download candidates into an isolated review directory.
.venv/bin/python scripts/prepare_public_images.py
open data/public-images/gallery.html
```

You can download only selected classes with, for example, `--labels udon_noodles shiitake_mushrooms beef_meatballs`. `--output PATH` selects another review directory. A repeat run skips unchanged successful downloads and retries failed ones. HTTP failures are reported rather than replaced with generated or incorrectly labeled images.

The script limits downloads to 20 MiB and decoded source photos to 60 megapixels. Saved images are oriented, resized to at most 1600 pixels per side, and re-encoded as JPEG without original metadata. `provenance.jsonl` preserves attribution, source and processed SHA256 hashes, changes, and pending review status. `gallery.html` displays the saved photos with their notes and source links. Nothing is imported into `data/samples.jsonl`; no model is trained or replaced.

Direct Commons image downloads are blocked by this cloud environment's network policy (HTTP 403). The downloader's offline behavior and processing are tested with explicit mock image bytes. Its live download behavior must be verified on a machine with Wikimedia access, such as your Mac.

## Review and import approved photos

Check the actual ingredient and composition in the gallery. For noodle pictures, verify thick white udon versus flat translucent starch noodles. Remove opaque packaging, mixed dishes, or crops with no clearly visible target. Some candidates explicitly need a crop. Do not import an entire source-caption collection without reviewing it.

Copy only approved photos into `data/public-reviewed/images`, retaining the same `<source_group>/<label>/filename.jpg` layout. Keep `provenance.jsonl` with this collection and record any additional crops in its `changes` field. Keep accepted/rejected decisions in its `review_status` field; copying a photo does not automatically update that record. The script does not certify review decisions.

Once approved photos are in that separate directory, explicitly import each photographer group as its own collection session:

```bash
for group in data/public-reviewed/images/*; do
  [ -d "$group" ] || continue
  .venv/bin/python scripts/import_images.py "$group" --session "$(basename "$group")"
done
```

Do not import the unreviewed `data/public-images/images` folder. A source group spans all photos by the same photographer, even across different ingredient classes. This conservative grouping prevents near-identical public photos from leaking between training and validation. The imported samples keep the group as `session_id`; keep the public provenance sidecar alongside the dataset because ordinary camera/export records do not include all attribution fields.

## Train once there is enough data

Add varied, correctly labeled images for all nine ingredients and background. Public photographs can supplement them; camera captures provide the held-up ingredient view the app will actually receive. Capture different backgrounds, angles, distances, hands, and separate sessions. A practical first collection target is 50–100 varied examples per ingredient, followed by failure-driven collection; the number alone does not establish accuracy.

The trainer's eight-image/two-session minimum is a workflow guard, not evidence that a model is well trained. The public starter alone intentionally cannot satisfy it for every class. Keep a fresh camera session out of both training and validation, then run:

```bash
.venv/bin/python scripts/prepare_model.py --mode mobilenet
.venv/bin/python scripts/train.py --epochs 15
.venv/bin/python scripts/evaluate.py --session camera-heldout-1
```

The evaluation command requires real samples labeled with that held-out session and rejects sessions already used in model training or validation. Review per-class results, the confusion matrix, rejection behavior and accepted-prediction coverage. Use those results to determine whether the new model improves on the baseline; no accuracy percentage is claimed for this collection.

## Image rights and sources

Every source entry links to its primary Commons file page and the applicable license. Retain the author, source link, license link, and change history with copied or cropped images. CC BY sources require attribution; CC BY-SA sources also require retaining their applicable share-alike terms for image adaptations. Source image licenses are separate from application code licensing.

FoodSeg103 has useful ingredient annotations, but its generic noodles, cabbage and tofu labels do not directly cover this vocabulary. Food-101 focuses on dishes. This starter therefore uses specific ingredient file pages rather than silently relabeling broad dataset classes. No synthetic images are included.
