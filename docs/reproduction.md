# LLS48 training and evaluation workflow

Start with LLS48 SwinB(224) or SwinB(384), for which matching spatial students
and MRTT weights are available in separate releases. The commands below connect
data preparation, feature extraction, MRTT training, and saved-prediction AP.
They are a source-derived workflow, **not a claim that a new image-based run
has reproduced the paper**. Use the published download manifests for unchanged
release assets. New training requires the explicit independent-validation
protocol described in [training protocol](training_protocol.md).

The two paths serve different purposes:

```text
Released student + LLS48 images -> keyed features -> released MRTT -> AP
New inner-split student -> new keyed features -> new MRTT training -> final test AP
```

Training a new spatial student is a separate multi-stage path described below.
It is not needed merely to use the released spatial encoder. LLS48 L224 has no
qualified spatial weight in this release; do not substitute B224/B384 features
or claim the paper ensemble is available end to end.

## Environment and downloads

Clone the repository and run commands from its root in an isolated Python 3.9
environment. Follow [installation](installation.md) and use the LLS48 spatial
profile, PyTorch 2.0.1 and timm 0.5.4, for feature extraction. Install
`requirements-spatial.txt`, then `python -m pip install -e ".[dev]"`.
The historical versions are compatibility targets, not a current security
support guarantee. Use a separate environment for CholecT45 L224.

After the corresponding release manifests report `published`:

```shell
curconmix-assets --asset lls48-dataset --destination data/LLS48 --install
python -m curconmix_release.lls48 verify --root data/LLS48
curconmix-assets --manifest manifests/spatial_assets.json --asset lls48-spatial-b224 --destination artifacts/spatial --install
curconmix-assets --asset lls48-mrtt-b224 --destination artifacts/checkpoints --install
```

Each weight bundle contains folds 0 through 4. Do not rename weights to another
fold or mix resolutions. Dataset images, annotation, mappings and released
weights must remain unchanged. See the [download guide](quickstart_lls48.md)
for disk space, interrupted downloads and offline checks.

## Prepare features with original frame identities

The opt-in feature exporter writes a fresh directory containing features,
labels, and a `feature_manifest.json`. The manifest binds feature and label
files to ordered original frame/clip keys and the spatial checkpoint identity.
It uses the packaged validation preprocessing and the student's pooled,
pre-classifier backbone features, not auxiliary predictions or contrastive
projections. No sigmoid is applied to features.

Use the command's default preview first. An execution command is an explicit
request for image decoding and spatial model forward passes. Those computations
were **not** run as part of preparing this workflow. See `python -m
curconmix_release.feature_export --help` for the exact options.

Create the parent `data/prepared` directory, then preview:

```shell
python -m curconmix_release.feature_export --config configs/temporal/lls48_base224_plus.json --asset lls48-spatial-b224 --dataset-root data/LLS48 --checkpoint-root artifacts/spatial --output data/prepared/lls48_b224
```

Add `--execute` only to intentionally compute the features. `--device cuda`
selects a compatible GPU; the default CPU path may be slow. This command requires
published, verified spatial weights and creates all five fold-specific feature
sets. It does not run MRTT, train a model or calculate AP.

The output layout is:

```text
data/prepared/lls48_b224/
  feature_manifest.json
  features/lls48/<model_name>/<fold>/train/VIDxx.npy
  features/lls48/<model_name>/<fold>/val/VIDxx.npy
  labels/lls48/triplet_label/VIDxx.npy
```

Every fold uses its own student for both its training and validation procedures.
The exporter preserves observed frame order and independent clip identities;
it does not interpolate images or features across unobserved gaps. An existing
output directory is not overwritten. A matching dimension and repeated GT do
not authenticate unrecorded legacy feature rows.

## Check prepared inputs

```shell
curconmix preflight --config configs/temporal/lls48_base224_plus.json --feature-root data/prepared/lls48_b224/features --label-root data/prepared/lls48_b224/labels --feature-manifest data/prepared/lls48_b224/feature_manifest.json
```

For independently prepared features without this manifest, the older array
preflight remains available, but it establishes only array/fold consistency,
not image identity. Do not attach a newly invented manifest to old arrays.

## Use the released MRTT

Preview inference without running a model:

```shell
curconmix infer-temporal --config configs/temporal/lls48_base224_plus.json --feature-root data/prepared/lls48_b224/features --label-root data/prepared/lls48_b224/labels --feature-manifest data/prepared/lls48_b224/feature_manifest.json --checkpoint-dir artifacts/checkpoints/lls48/b224 --output outputs/lls48_b224_released.csv --dry-run
```

Remove `--dry-run` only when intentionally executing inference with the qualified
inputs and runtime. Checkpoints are loaded with `weights_only=True` and strict
state matching; dataset, architecture and embedded fold must agree. With a
validated feature manifest, the output restores original frame/clip metadata
only after full label and row-order checks. Without it, synthetic sequential
IDs are not a clip-bounded evaluation manifest.

## Train a new MRTT

Training does not fine-tune the spatial student. It requires newly trained
inner-split student features, not the released-student features above. Supply
an explicit split and a version-2 training feature manifest. First preview:

```shell
curconmix train-temporal --config configs/temporal/lls48_base224_plus.json --feature-root data/new_features/features --label-root data/new_features/labels --training-split protocol/splits.json --training-feature-manifest data/new_features/training_feature_manifest.json --output outputs/lls48_b224_run --dry-run
```

Remove `--dry-run` only to start a new five-fold run. Use a new output directory.
The run records the explicit inner-split procedure order. To repeat that order on another machine,
pass `--procedure-order outputs/lls48_b224_run/procedure_order.json` to a **new**
run with the same inputs. This improves replayability but cannot reconstruct
unrecorded historical order or guarantee bitwise GPU determinism.

The new weights are saved under:

```text
outputs/lls48_b224_run/checkpoints/LLS48_mr2_Base224_Student_ks456_lr30e4_fusion_snorm_ksweight/best_model_fold0.pth
...
outputs/lls48_b224_run/checkpoints/LLS48_mr2_Base224_Student_ks456_lr30e4_fusion_snorm_ksweight/best_model_fold4.pth
```

Use that **experiment directory**, not its parent `checkpoints/`, with
`infer-temporal --checkpoint-dir`. The new serialization records an explicit
fold and plain Python metadata compatible with restricted loading. Historical
training checkpoints without fold metadata or with unsafe pickle objects are
not silently converted or loaded without safeguards.

## Evaluate six AP components

After an explicitly executed inference run creates the CSV:

```shell
curconmix evaluate-ap --dataset lls48 --predictions outputs/lls48_b224_released.csv --output outputs/lls48_b224_released_ap.json
```

This computes I, V, T, IV, IT and IVT AP from stored triplet probabilities and
GT with the existing decomposition, null handling and procedure aggregation.
It does not apply a second sigmoid or run a model. Values use the 0–1 scale;
multiply by 100 for percentage display. Five-fold SD is population SD
(`ddof=0`). A best-training validation score is not a replacement for this
six-component final-test report. Checkpoint selection uses inner-validation
AP only; final reporting is a separate, frozen-model outer-test evaluation.

LLS48 segmental F1 remains a separate **clip-bounded class-wise multi-label
segmental F1** calculation: reset event extraction for every five-frame clip.
Never concatenate clips or interpolate across the approximately 90-second
unsampled gaps. It measures local consistency, not dense transition localization
or continuity across gaps. The AP command does not compute F1.

## Use SwinB384

Replace all B224-specific choices together:

| Choice | SwinB224 | SwinB384 |
| --- | --- | --- |
| Spatial asset | `lls48-spatial-b224` | `lls48-spatial-b384` |
| MRTT asset | `lls48-mrtt-b224` | `lls48-mrtt-b384` |
| Temporal config | `lls48_base224_plus.json` | `lls48_base384_plus.json` |
| Released MRTT directory | `artifacts/checkpoints/lls48/b224` | `artifacts/checkpoints/lls48/b384` |
| Fresh feature output | `data/prepared/lls48_b224` | `data/prepared/lls48_b384` |

Both produce 1,024-dimensional features, so dimension checking alone cannot
detect a resolution/checkpoint mix-up. Keep the feature manifest and asset
identity checks enabled. Spatial **training** launchers remain B224-only.

## Train a new spatial B224 model

The spatial stages are cache preparation → contrastive pretraining → teacher
training → teacher soft labels → student training → saved predictions.
The cache generator ports the source's ImageNet-pretrained backbone feature
extraction and cosine-similarity recipe. It creates **new source-derived
caches**, not authenticated copies of historical caches. It is different from
the final-student feature exporter used by MRTT.

Preview and inspect the required data without importing a model:

```shell
python -m curconmix_release.cache_prepare --dataset lls48 --data-root data/LLS48 --training-split protocol/splits.json --output-dir data/lls48_b224_caches
python -m curconmix_release.cache_prepare --dataset lls48 --data-root data/LLS48 --training-split protocol/splits.json --output-dir data/lls48_b224_caches --check-only
```

Only when intentionally starting computation and permitting ImageNet pretrained
weight retrieval, add `--execute --allow-pretrained-download`. Similarity
matrices grow quadratically with training frames; inspect the storage estimate,
especially for CholecT45. Never substitute another fold's caches.

After generating the caches, check pretraining inputs:

```shell
python legacy/spatial/lls48/run_stage.py pretrain --data-root data/LLS48 --training-split protocol/splits.json --output-dir outputs/spatial_lls48_b224 --feature-dir data/lls48_b224_caches --check-only
```

Follow [the spatial stage guide](spatial.md#run-one-stage-at-a-time), replacing
its paths with these paths. Every `--execute` stage checks required input files
before importing the training implementation. New cache generation requires a
new directory; stage outputs share the same run directory across their sequence.
The final student outer-test prediction and evaluation are separate from the
training summary. New spatial training selects teacher and student checkpoints
only on inner-validation AP_IVT. Its selection logs must not be presented as
the best saved student's final outer-test AP or reproduced paper performance.

The historical optimizer, sampler and scheduler behavior is retained. In
particular, the configured accumulation value does not establish an effective
batch of twice the configured batch, and pretraining contains both batch and
epoch scheduler calls. A redesigned training recipe must be reported separately,
not silently substituted for the historical implementation.

## Validation boundary

Synthetic tests cover input failures, serialization, fold identity, order
replay, output collisions and keyed feature contracts. CI separately installs
the spatial dependency profile and checks imports and configuration composition.
Neither step trains a model or proves published AP/F1. Real image-to-feature
numeric comparison, released-weight inference and retraining performance remain
separate qualification tasks. See [validation](validation.md).
