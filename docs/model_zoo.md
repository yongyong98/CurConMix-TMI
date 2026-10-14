# Model availability

The [LLS48 v1.0.0 release](https://github.com/yongyong98/CurConMix-TMI/releases/tag/lls48-v1.0.0)
provides **15 model-only LLS48 MRTT checkpoints**, under **CC BY-NC 4.0**.
See the [quickstart](quickstart_lls48.md#4-download-an-mrtt-variant) for download
and installation. Weights live in GitHub Releases, not the Git source history.

This is a temporal-checkpoint release, **not a numerically qualified end-to-end
model zoo**. Precomputed verified feature arrays are not included. LLS48 B224/B384 and
CholecT45 L224 spatial students are available in a
[separate published release](https://github.com/yongyong98/CurConMix-TMI/releases/tag/spatial-encoders-v1.0.0).
See the [spatial checkpoint guide](spatial_checkpoints.md).

## MRTT configurations

CurConMix+ applies the Multi-Resolution Temporal Transformer (MRTT) to features
from a frozen CurConMix student encoder. The paper-aligned implementation is
`mr2` (`MultiResTemporalModel2`). The ensemble averages predictions from the
SwinB(384) and SwinL(224) variants; it has no separate ensemble weight file.

| Dataset | Spatial backbone | Feature dimension | Temporal scales | Public weights |
| --- | --- | ---: | --- | --- |
| CholecT45 | SwinB(224) | 1,024 | 4, 5, 6, 7 | Unpublished |
| CholecT45 | SwinB(384) | 1,024 | 4, 5, 6, 7 | Unpublished |
| CholecT45 | SwinL(224) | 1,536 | 4, 5, 6 | Unpublished |
| LLS48 | SwinB(224) | 1,024 | 4, 5, 6 | `lls48-mrtt-b224`, folds 0–4 |
| LLS48 | SwinB(384) | 1,024 | 4, 5, 6 | `lls48-mrtt-b384`, folds 0–4 |
| LLS48 | SwinL(224) | 1,536 | 4, 5, 6, 7 | `lls48-mrtt-l224`, folds 0–4 |

## What was verified

The released files were derived from the exact five-fold checkpoint paths in the
canonical model registry, not selected by filename recency or a newly measured
score. Original files were preserved and SHA-256 checked before and after export.
The exports contain model tensors, complete architecture metadata, dataset,
variant, fold, mapping checksum, source-file checksum, and asset license. They do
not contain optimizer/scheduler state, training arguments, or private paths.

For all 15 files, tensor keys, order, shapes, dtypes, and values match the originals
exactly. The public MRTT implementation strict-loads all exports. The public
checkpoint/meta/model-loading path was tested in PyTorch 2.0.1 and 2.8 using
`weights_only=True`, without forward passes. Original architecture metadata also
matches the canonical sidecars and public configurations.

Fold identity follows the canonical registry's exact `best_model_foldN.pth`
paths. The originals do not contain an independent embedded fold ID; the export
records this provenance basis rather than claiming one was recovered.

These checks establish export equivalence and loading compatibility. They do
**not** constitute a new inference run or reproduction of the paper's AP/F1.

## Spatial encoder dependency

An MRTT checkpoint alone cannot process raw images. Image-based inference also
requires the matching spatial student encoder, preprocessing, class mappings,
fold assignment, and feature-extraction procedure. LLS48 B224/B384 and CholecT45
L224 students have
exact stored-prediction pairing across all five folds and are publicly available.
LLS48 L224 is still missing from the audited checkpoint storage; the
complete spatial student set is not available. See the
[spatial checkpoint guide](spatial_checkpoints.md) for scope and loading checks.

Candidate stored LLS48 feature arrays passed dimension, finite-value, fold, and
full-GT checks. They did not include image-row keys or extraction provenance that
establishes exact frame identity. Since labels repeat within five-frame clips,
GT equality cannot prove feature-row order. These candidate arrays are therefore
not distributed as verified features in this release.

The new [B224/B384 workflow](reproduction.md) can explicitly produce features,
labels and original-key manifests from the matching released spatial students.
This source-derived exporter has not been numerically compared to historical
features or the paper's predictions. Feature-based execution also remains
available for users who already have independently qualified inputs.
See the [quickstart](quickstart_lls48.md) and
[spatial implementation guide](spatial.md). Separate phase, step, and task
temporal-model weights are not part of this release.

## Checkpoint security and terms

Download only from the official release, verify the manifest SHA-256, and use
restricted/weights-only loading. The exported checkpoints contain tensors and
plain Python metadata; no additional custom or NumPy allowlist is required to
load them. Never disable loading safeguards to open an untrusted checkpoint.
A checksum establishes integrity, not the trustworthiness of an unknown source.

The historical runtime is not a security-support guarantee. See
[installation](installation.md) and [data/model terms](../DATA_AND_MODEL_LICENSE.md).
The models are research resources, not clinically validated medical devices.
