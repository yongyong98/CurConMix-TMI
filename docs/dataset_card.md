# LLS48 dataset card

## Scope

LLS48 contains 48 laparoscopic liver surgery procedures represented by 1,912
independently sampled five-frame clips, totaling 9,560 annotated PNG images.
The release includes exactly the images selected by `dataframe/LLS48_TRI195.csv`.
Image bytes, relative paths, annotation row order, labels, and class names are
preserved. Additional images from the source collection are not included.

The action taxonomy has 195 triplets, 17 instruments, 16 verbs, and 38 targets.
The unchanged annotation also includes step indices 0–4 and task indices 0–17.
This release does not invent textual step/task class names where no verified
class-name dictionary accompanies the action benchmark.

## Files

All archives extract into a single dataset root without an extra enclosing folder:

```text
LLS48/
  data/VID01/01_001/frame_00001.png
  dataframe/LLS48_TRI195.csv
  dict/instrument_class_mapping.txt
  dict/verb_class_mapping.txt
  dict/target_class_mapping.txt
  dict/triplet_class_mapping.txt
  splits/folds.json
  manifests/frames.json
  manifests/checksums.json
  DATASET_CARD.md
  DATA_AND_MODEL_LICENSE.md
```

`video` is the pseudonymous procedure identifier. `clip_name` identifies an
independent clip. `image_path` is relative to `data/`, and `image_id` is a stored
image key. The index-list columns are retained from the source. Binary columns
`tri0`–`tri194`, `inst0`–`inst16`, `v0`–`v15`, and `t0`–`t37` retain their original
positions. The CSV is copied byte-for-byte, not regenerated.

`frames.json` records the CSV row index, original identifiers, within-clip frame
index, and held-out fold. `folds.json` preserves the existing five procedure-level
cross-validation folds (0–4). Every frame and clip from a procedure stays in the
same fold. Do not randomly split frames or clips across training and test sets.

The mapping files preserve the original taxonomy, including separate instrument
classes named `bulldog` and `bulldog clamp` and the target named `bulldog`.
Do not merge or rename these classes when evaluating the published checkpoints.
The four dictionaries are checked against the evaluator's numeric triplet map.
Older alternative dictionaries are not part of this release.

## Integrity

The canonical annotation SHA-256 is
`6835eb20fb1272088fdf263a9c417dbc7bd56e0a4e5c5580f513496ca947af50`.
The selected image payload totals 26,410,486,025 bytes before archive headers.
The images are distributed in independently extractable ZIP shards below 1 GiB,
with no resizing, transcoding, or intensity changes. The metadata archive contains
the per-file SHA-256 checksums. Downloads additionally verify archive sizes and
SHA-256 values before installation. Extraction and verification require additional
disk space beyond the retained download archives.

After installation, verify every image and the complete annotation selection:

```bash
python -m curconmix_release.lls48 verify --root data/LLS48
```

Verification reads all image bytes and can take several minutes. Archive integrity
checks establish byte consistency, not new model-performance measurements.

## Temporal interpretation

LLS48 is not a complete, densely annotated video timeline. Clips are separated by
approximately 90-second unsampled gaps. For clip-bounded class-wise multi-label
segmental F1, each five-frame clip is a separate temporal unit. Event extraction
resets at each clip boundary. Never concatenate clips or interpolate events across
the gaps. This metric supports local temporal consistency within annotated clips,
not dense transition localization, action persistence across gaps, or
continuous-clock long-range reasoning.

Action AP uses the existing triplet-to-component decomposition and procedure-wise
AP aggregation. That aggregation does not make segmental F1 a continuous
procedure-level temporal evaluation. Preserve the existing null handling and
probability semantics; stored probabilities must not receive a second sigmoid.

## Access, license, and privacy

The authors confirm institutional approval and de-identification for public
release of the selected images and annotations. This packaging process preserves
the approved image bytes; it does not substitute for institutional privacy review.
The metadata uses pseudonymous procedure and clip identifiers and relative file
paths, without patient names or internal storage paths.

LLS48 images and annotations are provided under CC BY-NC 4.0. Attribute the
CurConMix+ authors and dataset when reusing them. Commercial reuse requires
separate permission. Do not attempt to reidentify patients. The repository's MIT
code license does not apply to these data or to model checkpoints. See
`DATA_AND_MODEL_LICENSE.md` for the release terms and license link.

The data and models are research resources, not a validated clinical device.
