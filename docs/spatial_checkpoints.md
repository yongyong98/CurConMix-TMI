# Spatial student checkpoints

CurConMix is the image encoder; MRTT is the temporal model added by CurConMix+.
Their weights are different assets. The published
[spatial encoder release](https://github.com/yongyong98/CurConMix-TMI/releases/tag/spatial-encoders-v1.0.0)
is separate from the LLS48 dataset/MRTT release so that each download has its own
verified manifest.

## Release scope

| Dataset / student | Spatial release scope |
| --- | --- |
| LLS48 SwinB(224) | Published, five folds, 0–4 |
| LLS48 SwinB(384) | Published, five folds, 0–4 |
| LLS48 SwinL(224) | Not available: matching source checkpoints have not been located in the audited storage |
| CholecT45 SwinB(224) | Historical MICCAI weights remain at the original location below |
| CholecT45 SwinB(384) | Not included: complete five-fold checkpoint identity is not established |
| CholecT45 SwinL(224) | Published, five folds, 0–4 |

The three bundles are `lls48-spatial-b224`, `lls48-spatial-b384`, and
`cholect45-spatial-l224` in
[`manifests/spatial_assets.json`](../manifests/spatial_assets.json). Release tag:
`spatial-encoders-v1.0.0`. These exports use **CC BY-NC 4.0**, independently of
the MIT source-code license. No pretraining or teacher weights are implied.

## Download and inspect

From a clone of this repository, install the base package for downloading.
Use separate model environments because the qualified timm versions differ:

| Spatial bundle | Qualified PyTorch | Required timm | Exact model name |
| --- | --- | --- | --- |
| LLS48 B224 | 2.0.1 | 0.5.4 | `swin_base_patch4_window7_224` |
| LLS48 B384 | 2.0.1 | 0.5.4 | `swin_base_patch4_window12_384` |
| CholecT45 L224 | 2.0.1 | 1.0.20 | `swin_large_patch4_window7_224.ms_in22k` |

These are verified loading environments, not proof of the original training
versions or a security-support guarantee. The [spatial guide](spatial.md)'s
requirements pin timm 0.5.4 for LLS48 loading; for CholecT45 L224, use a separate
environment with timm 1.0.20. Do not remap state keys or use `strict=False` to
force a checkpoint into the other architecture version. Installing the package
does not download weights.

```shell
curconmix-assets --manifest manifests/spatial_assets.json --list
curconmix-assets --manifest manifests/spatial_assets.json --asset lls48-spatial-b224 --destination artifacts/spatial --install
python -m curconmix_release.spatial_checkpoint --asset lls48-spatial-b224 --root artifacts/spatial --fold 0
```

Use `lls48-spatial-b384` or `cholect45-spatial-l224` for the other bundles.
Each bundle contains all five folds. Keep the spatial and MRTT fold and backbone
consistent. The downloader refuses unpublished assets and verifies file sizes
and SHA-256. It never overwrites a differing installed file. The inspection
command verifies the file hash before
restricted loading and strictly loads the architecture with `pretrained=False`.
It does not download ImageNet weights, run a forward pass, or generate features.

To intentionally compute LLS48 B224/B384 features, use the separate
[feature and evaluation workflow](reproduction.md). Its exporter preserves frame
and clip identity and checks fold-matched weights, but numerical equivalence to
historical features and published AP remains unqualified. Do not rename these
model-only files to masquerade as a historical pretraining/teacher checkpoint.

## What the checks establish

For each included fold, the original checkpoint's embedded prediction
array matches its canonical stored prediction CSV element-for-element after
conversion to the original float32 precision. The historical training code saves
the model state and validation predictions together. This establishes stored
checkpoint/prediction pairing; it is **not a newly executed inference test**.

The distribution contains only unchanged model tensors and plain metadata.
Optimizer state, prediction arrays, private paths, and training arguments are
excluded. Source hashes are checked before and after export. Tensor keys,
shapes, dtypes, and values must be identical, and the historical public model
must strictly load each export without a forward pass. The release records
dataset, fold, architecture, image size, class mapping hash, and source hash.

These checks do not establish that previously stored feature arrays were
extracted from these exact weights or preserve exact image-row order. A
qualified image-to-MRTT feature exporter and clean-machine end-to-end metric
reproduction remain outstanding. In particular, the legacy similarity-cache
generator is not a validated final-student feature exporter.

LLS48 segmental evaluation remains **clip-bounded**: independently sampled
five-frame clips are separate event-extraction units. Do not interpolate or
claim continuity across the unsampled gaps.

## Existing MICCAI resources

The original [MIDAS-SurgAI/CurConMix repository](https://github.com/MIDAS-SurgAI/CurConMix)
and its [OneDrive checkpoint folder](https://1drv.ms/f/c/d6832cd62003d47f/EnWJati1F9JBgP68ruh3k2YBHaLC_yoZfeNxTBI4KMet4Q?e=HVrcdE)
are preserved. The visible folder contains five CholecT45 B224 pretraining and
five B224 student checkpoints, plus the CholecT45 annotation CSV. This is a
historical resource, not a claim of byte identity with every TMI checkpoint.
The new asset license is not retroactively assigned to those externally hosted
files. Consult their existing terms and the authors before reuse.

Readers can follow the original repository's TMI notice to this repository.
No manuscript URL change, repository rename, or private-repository disclosure
is required for that navigation path.
