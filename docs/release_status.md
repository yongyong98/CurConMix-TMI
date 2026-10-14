# Release status

The [LLS48 v1.0.0 release](https://github.com/yongyong98/CurConMix-TMI/releases/tag/lls48-v1.0.0)
provides the dataset and integrity-checked temporal weights. The independent
[spatial encoder release](https://github.com/yongyong98/CurConMix-TMI/releases/tag/spatial-encoders-v1.0.0)
provides the verified LLS48 B224/B384 and CholecT45 L224 student bundles.
Data/checkpoint availability must not be confused with reproducing the paper's
experiments from raw images.

| Item | Status |
| --- | --- |
| Independent source tree | Public, without private Git history |
| MRTT model and metric source | Portable training, inference, and AP entry points |
| Spatial research source | Separate legacy implementation; see spatial guide |
| LLS48 images and annotations | 9,560 canonical images, unchanged CSV, mappings, folds, checksums; 26 ZIPs |
| LLS48 MRTT weights | Model-only B224/B384/L224, five folds each; all 15 tensor-equivalent and strict-load tested |
| CholecT45 MRTT weights | Not published in this release |
| Spatial student weights | Published LLS48 B224/B384 and CholecT45 L224, five folds each; [scope and checks](spatial_checkpoints.md) |
| New keyed feature workflow | Source-derived LLS48 B224/B384 exporter, labels and original-key checks; numerical behavior not yet qualified |
| Spatial cache preparation | Explicit B224 source-derived cache generation and stage input checks; no bit-identical historical cache claim |
| End-to-end clean-machine training | Not run or qualified |
| Paper-result reproduction from released weights | Not run or qualified |
| Licenses | Author-owned code MIT; LLS48 and released weights CC BY-NC 4.0; upstream licenses retained |

## Storage and scientific boundary

Large data and weights are hosted as GitHub Release assets, not committed to
Git. The source repository contains download manifests with exact sizes and
SHA-256 values. The installer validates downloads, does not overwrite differing
files, and supports verified part-level retries and offline installed-file checks.
See the [quickstart](quickstart_lls48.md).

LLS48 annotation bytes, original PNGs, class taxonomy, frame order, and
procedure-level folds are preserved. Additional unannotated-selection images,
private review materials, training states, and internal storage paths are excluded.
Institutional approval and de-identification are author-confirmed; packaging is
not an independent clinical privacy certification.

LLS48 segmental evaluation is **clip-bounded**, with event extraction reset for
each independently sampled five-frame clip. It supports local temporal
consistency, not dense transitions or continuity through unsampled gaps.

## Remaining work before a complete image pipeline

1. Numerically qualify the new source-derived feature exporter and cache
   preparation path. Original keys and hashes are software-checked, but a real
   image-based reproduction has not been run.
2. Validate clean-machine image-based execution and measure released-model
   predictions against the published metrics, separately from software tests.

Missing checkpoint variants remain unavailable; other backbones or folds must
not be substituted for them.

No training or inference was performed to prepare this release. See
[validation](validation.md) for exactly what was checked.

## Lab visibility

The original public [MIDAS-SurgAI/CurConMix repository](https://github.com/MIDAS-SurgAI/CurConMix)
now links to this TMI companion and its releases. Its MICCAI content and existing
OneDrive links are preserved. This keeps the original web address useful without
editing the manuscript or disclosing the internal private repository.

The lab GitHub identity is a user account, not an organization. The README link
does not transfer repository ownership. The lab account owner can additionally
fork this personal public repository and pin it on that profile.
