# CurConMix+

[![CPU tests](https://github.com/yongyong98/CurConMix-TMI/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/yongyong98/CurConMix-TMI/actions/workflows/cpu-tests.yml)

Research code for CurConMix+, accepted for publication in IEEE Transactions on
Medical Imaging. CurConMix learns spatial representations with curriculum-guided
contrastive learning and feature mixup. CurConMix+ adds the Multi-Resolution
Temporal Transformer (MRTT) for action recognition.

**Published LLS48 data and MRTT release:** the download provides the 9,560-image
benchmark, unchanged annotations, five-fold splits, and 15 LLS48 temporal
checkpoints with verified checksums in
[LLS48 v1.0.0](https://github.com/yongyong98/CurConMix-TMI/releases/tag/lls48-v1.0.0).
See the [LLS48 quickstart](docs/quickstart_lls48.md).
LLS48 B224/B384 and CholecT45 L224 spatial students are available in the separate
[spatial encoder release](https://github.com/yongyong98/CurConMix-TMI/releases/tag/spatial-encoders-v1.0.0);
see the [spatial checkpoint guide](docs/spatial_checkpoints.md). A source-derived
LLS48 B224/B384 feature exporter and spatial cache preparation commands are now
included. They preserve input identities and require explicit execution; their
image-based numerical behavior and paper-score reproduction are not yet qualified.
See [release status](docs/release_status.md).

Research group: [MIDAS-SurgAI](https://github.com/MIDAS-SurgAI).
Repository: [yongyong98/CurConMix-TMI](https://github.com/yongyong98/CurConMix-TMI).

## Start here

| Goal | Entry point | Current boundary |
| --- | --- | --- |
| Evaluate saved action probabilities | `curconmix evaluate-ap` | Requires an authorized prediction CSV with GT |
| Check prepared temporal features | `curconmix preflight` | Validates arrays and folds, not source-image identity |
| Train MRTT from prepared features | `curconmix train-temporal` | Explicit inner-validation split and new-encoder feature provenance required |
| Run MRTT checkpoint inference | `curconmix infer-temporal` | Explicit action; matching features and metadata required |
| Inspect spatial training code | [Spatial guide](docs/spatial.md) | Historical multi-stage implementation; not turnkey |
| Access LLS48 | [Download and verify](docs/quickstart_lls48.md) | 9,560 selected images, annotations, mappings, folds |
| Access model weights | [Model availability](docs/model_zoo.md) | Published LLS48 MRTT and spatial bundles; see the per-dataset scope |
| Connect images, features, training and AP | [Training and evaluation workflow](docs/reproduction.md) | B224/B384 source-derived feature bridge; no end-to-end paper-score claim |

This action-recognition package does not contain a qualified release of separate
phase, step, or task temporal models.

New training uses [independent inner-validation checkpoint selection](docs/training_protocol.md).
The published outer folds are reserved for final testing. Released checkpoints
remain usable for inference; new training results must be reported separately.

## Install and test

Use an isolated Python 3.9 environment. See [installation](docs/installation.md)
for setup and the historical PyTorch version needed for model execution.

```shell
python -m pip install -r requirements-temporal.txt
python -m pip install -e ".[dev]"
curconmix --help
curconmix-assets --list
python -m pytest -q
```

The help command, asset listing, and CPU tests do not launch training or
checkpoint inference. Installation does not download datasets or weights.

## Download LLS48 and MRTT weights

```shell
curconmix-assets --asset lls48-dataset --destination data/LLS48 --install
python -m curconmix_release.lls48 verify --root data/LLS48
curconmix-assets --asset lls48-mrtt-b224 --destination artifacts/checkpoints --install
```

Allow 55 GiB for the dataset, 75 GiB for the dataset and all LLS48 MRTT variants,
or 90 GiB when also keeping all three spatial bundles, including retained
download caches. Interrupted downloads resume at verified
part boundaries. See the [quickstart](docs/quickstart_lls48.md) for cache placement,
offline verification, a frame/clip reader, and the feature-based execution boundary.

## Evaluate saved predictions

```shell
curconmix evaluate-ap --dataset lls48 --predictions outputs/lls48.csv --output outputs/lls48_ap.json
```

The evaluator reads probabilities and GT from the input CSV and calculates all
six AP components: I, V, T, IV, IT, and IVT. It preserves the dataset-specific
triplet decomposition, null-triplet handling, and video/procedure AP aggregation.
No extra sigmoid is applied to stored probabilities. Five-fold mean and standard
deviation use the canonical `nanmean` and population `nanstd` (`ddof=0`). Raw AP
values are on a 0–1 scale; percentage display values multiply by 100.

LLS48 segmental F1 is a separate **clip-bounded** metric: independently sampled
five-frame clips are never joined across unsampled gaps for event extraction.
It measures local consistency, not dense transition localization or continuity
through those gaps. See [data and metric boundaries](docs/data.md). The AP command
does not report F1 or claim to have recomputed it.

## Prepare feature-based execution

The six dataset/backbone configurations are in `configs/temporal/`. Point the
commands to your own authorized feature and label directories. No private
storage path is required.

```text
data/features/<dataset>/<model_name>/<fold>/train/<video>.npy
data/features/<dataset>/<model_name>/<fold>/val/<video>.npy
data/labels/<dataset>/triplet_label/<video>.npy
```

Features have shape `[frames, feature_dimension]`. Labels have shape
`[frames, total_action_classes]`, with triplets followed by instrument, verb, and
target classes. CholecT45 uses 131 total classes; LLS48 uses 266. Frame order must
match exactly. Array checks cannot establish image provenance or replace a
validated spatial-feature exporter.

```shell
curconmix preflight --config configs/temporal/lls48_base224_plus.json --feature-root data/features --label-root data/labels
```

Preview the explicit training command without running it:

```shell
curconmix train-temporal --config configs/temporal/lls48_base224_plus.json --feature-root data/features --label-root data/labels --training-split protocol/splits.json --training-feature-manifest data/training_feature_manifest.json --output outputs/lls48_b224 --dry-run
```

Preview checkpoint inference independently of training:

```shell
curconmix infer-temporal --config configs/temporal/lls48_base224_plus.json --feature-root data/features --label-root data/labels --checkpoint-dir artifacts/checkpoints/lls48/b224 --output outputs/lls48_b224.csv --dry-run
```

Remove `--dry-run` only when intentionally executing that stage with qualified
inputs and a compatible runtime. MRTT weights alone cannot process raw images;
the matching student encoder and feature-extraction procedure are also required.

With `--feature-manifest`, the LLS48 B224/B384 workflow validates keyed feature
files and restores original frame/clip metadata after full-GT checks. Without
that manifest, inference uses synthetic sequential row IDs and does not preserve
original clip identities. Do not use those synthetic IDs for clip-bounded F1 or
reconstruct clip boundaries from adjacency alone. The temporal model consumes
procedure-ordered observed features; only segmental evaluation is clip-bounded.
See the [complete command sequence](docs/reproduction.md), including spatial
cache preparation and the exact new-training checkpoint directory.

## Layout

```text
configs/temporal/       Dataset and backbone MRTT settings
src/curconmix_repro/    MRTT, canonical metrics, and separate command entry points
src/curconmix_release/  Explicit checksum-verified asset download utility
legacy/spatial/        Dataset-specific historical CurConMix implementations
manifests/             Public asset availability, URLs, and checksums
docs/                  Installation, data, models, and release boundaries
tests/                 CPU tests using synthetic fixtures
third_party/           Retained upstream licenses
```

## Licensing and acknowledgments

The author-owned source code is licensed under the [MIT License](LICENSE).
LLS48 images/annotations
and released checkpoints use CC BY-NC 4.0 (noncommercial use); see
[data and model terms](DATA_AND_MODEL_LICENSE.md). These terms do not turn
unpublished assets into available downloads. Third-party code keeps its original
license; see [third-party notices](THIRD_PARTY_NOTICES.md).

CholecT45 and the action-triplet metrics build on
[CAMMA's CholecT45](https://github.com/CAMMA-public/cholect45) and
[ivtmetrics](https://github.com/CAMMA-public/ivtmetrics). Supervised contrastive
losses build on [SupContrast](https://github.com/HobbitLong/SupContrast).

The final TMI citation and DOI will be added when confirmed. Do not infer a DOI,
volume, or page range from the repository name. This research implementation is
not a clinically validated medical device.
