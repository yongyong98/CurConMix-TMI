# Spatial CurConMix training

The spatial encoder is trained in three stages: curriculum-guided contrastive pretraining, supervised teacher training, and student training with teacher soft labels. The source supports CholecT45 and LLS48 with explicit SwinB(224) stage configurations and a new independent-validation protocol. Approved data, an investigator-specified inner split, matching new caches, and a compatible runtime are required. These commands do not establish end-to-end reproduction of the published results; released checkpoints and predictions are unchanged.

## Support scope

For stored final-student weights, see the separate
[spatial checkpoint guide](spatial_checkpoints.md). Its B224/B384 inspection
support does not expand the B224-only training launcher described below.

| Capability | Availability |
| --- | --- |
| CholecT45 and LLS48 SwinB(224) spatial source | Included under `legacy/spatial` |
| Original model, loss, augmentation, canonical outer folds, and AP implementations | Preserved; new training adds a separate inner-validation partition |
| Stage command preview | Available without importing PyTorch or reading data |
| Full training and checkpoint inference | Requires a prepared dataset and compatible environment; not release-validated end to end |
| Contrastive feature and similarity cache creation | Opt-in B224 source-derived preparation command; not proven bit-identical to historical caches |
| SwinB(384), SwinL(224), challenge splits, DINO, and linear probing | Not supported by the packaged spatial launcher |
| Step and task models | Not part of this action-level spatial package |

Do not interpret a successful command preview or syntax check as a reproduced paper result. Use the main repository evaluation interface for saved-prediction AP evaluation. The dataset-specific `evaluate.py` files remain available as historical implementations.

## Environment

Use a separate Python 3.9 environment for the legacy spatial code. `requirements-spatial.txt` is a minimal historical API compatibility profile, not a fully resolved or freshly validated environment lock. It retains PyTorch 2.0.1, torchvision 0.15.2, timm 0.5.4, and Albumentations 1.0.3. The entry points use Hydra 1.3.2 to keep paths explicit and prevent changes of working directory. The profile is limited to the B224 model name; do not assume that newer timm model aliases are available.

Install a PyTorch build appropriate for your CUDA environment, then install the spatial requirements:

```bash
python -m pip install -r requirements-spatial.txt
python -m pip install -e "."
```

Training initializes the Swin backbone from timm pretrained weights. The first execution may download those weights unless they are already cached. No model is constructed or downloaded by the command-preview launcher. Neptune and W&B logging are disabled by default; no account credentials are supplied.

## Dataset and cache layout

Set the data root to your approved local dataset. Each annotation `image_path` is resolved below `<data-root>/<train_path>`, where `train_path` defaults to `data/`. Preserve the released annotation order and schema rather than regenerating class indices.

The CholecT45 loader selects the canonical 45-video split from a compatible annotation CSV. Its historical default CSV name is `dataframes/CholecT50.csv`; this filename does not change the explicit `cholect45-crossval` split. Use `--annotation-csv` to point to your authorized equivalent file. The LLS48 default is `dataframe/LLS48_TRI195.csv`.

The expected multi-label blocks are:

| Dataset | Triplet block | Instrument block | Verb block | Target block | Total outputs |
| --- | --- | --- | --- | --- | --- |
| CholecT45 | `tri0` through `tri99` | `inst0` through `inst5` | `v0` through `v9` | `t0` through `t14` | 131 |
| LLS48 | `tri0` through `tri194` | `inst0` through `inst16` | `v0` through `v15` | `t0` through `t37` | 266 |

The training loader uses the contiguous label block beginning at `tri0`. Do not reorder these columns. AP evaluation decomposes triplet predictions into component scores; auxiliary component heads are not substitutes for that decomposition.

Contrastive pretraining requires a separate trusted cache directory. With the default `--model-tag SwinB224`, it must contain these two files for each fold 0 through 4:

```text
SwinB224_fold0_trainset_features_threshold_10.pkl
SwinB224_fold_0_similarity_matrix.pkl
...
SwinB224_fold4_trainset_features_threshold_10.pkl
SwinB224_fold_4_similarity_matrix.pkl
```

Set `--model-tag` to the exact prefix of your verified caches. Each cache must correspond to that fold, backbone, dataset, explicit training split, and training frame order. A complete `cache_provenance.json` is mandatory; an `INCOMPLETE.json` marker blocks training. The launcher checks the annotation and split digests, ordered training-frame identities, source-image digests, and cache-file hashes. Filenames alone do not establish alignment. Historical caches lacking this provenance are not accepted for new training. Pickle caches must come from a trusted source because loading them can execute code.

For new B224 caches, use `python -m curconmix_release.cache_prepare` as described
in the [connected workflow](reproduction.md#train-a-new-spatial-b224-model).
It uses the source-derived ImageNet backbone and cosine-similarity recipe, not
the final-student MRTT feature exporter. The default is a preview; `--check-only`
checks inputs, while `--execute --allow-pretrained-download` explicitly permits
feature computation and pretrained weight retrieval. The output and provenance
belong to a new run, not to an authenticated historical cache. Pass the same
`--training-split /absolute/path/to/split.json` used for all later stages.

## Required independent training split

Create an explicit procedure-level split JSON following the
[training protocol](training_protocol.md). No inner-validation procedures are
chosen automatically. Each of the five folds must contain nonempty, disjoint
`train`, `inner_validation`, and `outer_test` lists whose union is the canonical
dataset. `outer_test` must exactly match the published outer fold. The manifest
also identifies the dataset and exact annotation CSV SHA-256. Optional patient
group identities are checked for overlap when supplied; procedure separation
alone does not establish patient separation without those identities.

All optimizer updates, contrastive caches, and teacher soft labels use `train`
only. Teacher and student checkpoint selection uses the existing AP_IVT metric
on `inner_validation` only. Outer-test predictions and final AP are generated
separately after training. Undefined inner-validation AP is an error, not a
fallback to test scores or the last epoch; ties retain the first best checkpoint.

New checkpoints carry embedded split metadata plus a
`<checkpoint>.provenance.json` sidecar with the actual checkpoint hash. Warm
starts and soft labels must match the split, annotation, fold, and stage.
Released historical weights without this lineage remain available for inference
but cannot be relabeled as valid warm starts for this new protocol. Fresh
ImageNet initialization is still supported. Do not fabricate provenance for old
weights, caches, or predictions.

## Stage settings

| Setting | CholecT45 B224 | LLS48 B224 |
| --- | --- | --- |
| Backbone | `swin_base_patch4_window7_224` | `swin_base_patch4_window7_224` |
| Image size | 224 × 224 | 224 × 224 |
| Pretraining epochs | 3 | 7 |
| Teacher epochs | 20 | 20 |
| Student epochs | 40 | 40 |
| Optimizer | AdamW | AdamW |
| Initial learning rate | 2 × 10⁻⁴ | 2 × 10⁻⁴ |
| Minimum learning rate | 2 × 10⁻⁵ | 2 × 10⁻⁵ |
| Weight decay | 1 × 10⁻⁶ | 1 × 10⁻⁶ |
| Configured batch size | 64 | 64 |
| Gradient accumulation steps | 2 | 2 |
| Mixup alpha | 0.4 | 0.4 |
| Contrastive temperature | 0.1 | 0.1 |
| Curriculum order | target → instrument–target → triplet | target → instrument–target → triplet |
| Seed | 142 | 142 |

These stage configurations retain the historical B224 hyperparameters, but new training uses the explicit independent-validation split. Contrastive pretraining has a fixed schedule: the final epoch is used without validation- or test-based selection. Teacher and student checkpoints are selected on inner-validation AP_IVT. This is a new training protocol, not evidence that the original runs used it or that retraining produces identical paper checkpoints. The contrastive sampler also controls how cached feature examples are assembled; the batch size is not a claim about the number of independent contrastive examples. The original optimizer and scheduler code is retained.

The historical mixup loop clears gradients every minibatch while stepping at the
configured accumulation interval; do not describe its accumulation value as
doubling the effective batch. Contrastive pretraining has both batch and epoch
scheduler calls. These semantics are retained for transparency, not silently
redesigned as part of the release tooling.

Training augmentations retain resize, horizontal flip, shift/scale/rotation, small hue/saturation/value perturbations, Gaussian noise, and ImageNet normalization. Validation uses resize and normalization. Cutout is disabled in the supplied configurations.

## Run one stage at a time

Run the following from the repository root. Replace the absolute paths with your own. For LLS48, change `cholect45` to `lls48`; the launcher then selects the LLS48 configuration and seven-epoch pretraining stage.

Preview a pretraining command without loading any model or dataset:

```bash
python legacy/spatial/cholect45/run_stage.py pretrain --data-root /absolute/path/to/dataset --training-split /absolute/path/to/split.json --output-dir /absolute/path/to/run --feature-dir /absolute/path/to/caches
```

Add `--check-only` to validate the annotation, image files and all stage-specific
cache/checkpoint/soft-label files and their split-bound provenance without
importing a training model. `--execute` performs the same checks before launching
the stage. Missing splits or incompatible artifacts block execution. Passing
these checks does not establish the trustworthiness of pickle contents or a
reproduced paper score.

To execute a stage, add `--execute`. Use the same output directory throughout the sequence:

```bash
python legacy/spatial/cholect45/run_stage.py pretrain --data-root /absolute/path/to/dataset --training-split /absolute/path/to/split.json --output-dir /absolute/path/to/run --feature-dir /absolute/path/to/caches --execute
python legacy/spatial/cholect45/run_stage.py teacher --data-root /absolute/path/to/dataset --training-split /absolute/path/to/split.json --output-dir /absolute/path/to/run --execute
python legacy/spatial/cholect45/run_stage.py softlabels --data-root /absolute/path/to/dataset --training-split /absolute/path/to/split.json --output-dir /absolute/path/to/run --execute
python legacy/spatial/cholect45/run_stage.py student --data-root /absolute/path/to/dataset --training-split /absolute/path/to/split.json --output-dir /absolute/path/to/run --execute
python legacy/spatial/cholect45/run_stage.py predict --data-root /absolute/path/to/dataset --training-split /absolute/path/to/split.json --output-dir /absolute/path/to/run --execute
```

Each stage covers folds 0 through 4 using the same explicit nested split. The teacher loads fixed final pretraining epoch 2 for CholecT45 or epoch 6 for LLS48, using zero-based checkpoint suffixes. The soft-label stage runs the selected teacher on that fold's `train` procedures only; it excludes both inner validation and outer test. It is a separate, necessary step. For an unchanged legacy released-checkpoint prediction workflow, omit `--training-split`; this preserves the original outer-fold inference mapping and does not certify the historical training protocol.

Evaluate saved best-student outer-test predictions separately with the public
six-component AP entry point. Training logs describe inner-validation checkpoint
selection, not final outer-test results.

Use a new output directory for each run, shared across that run's stages. New training refuses existing stage checkpoints, and soft-label generation refuses existing soft-label outputs; an interrupted run is not silently resumed. Keep final prediction destinations separate from prior results as well. Outputs include `checkpoints/` with provenance sidecars, `softlabels/` with provenance sidecars, `predictions/`, and training summaries. The packaged experiment names are `CurConMix_B224_Pretrain`, `CurConMix_B224_Teacher`, and `CurConMix_B224_Student`; they are names for new runs, not aliases that authenticate a released checkpoint.

## LLS48 interpretation

LLS48 contains independently sampled five-frame clips. A shared procedure identifier does not make adjacent clips a continuously observed video. For clip-bounded action segmental evaluation, reset event extraction at every clip boundary and never interpolate across unsampled gaps. These scores measure local temporal consistency within annotated clips, not dense transition localization or continuity across gaps. Spatial training and AP evaluation do not remove that interpretation boundary.
