# LLS48: download, verify, and use

The [LLS48 v1.0.0 release](https://github.com/yongyong98/CurConMix-TMI/releases/tag/lls48-v1.0.0)
provides the annotation-selected dataset and 15 model-only MRTT checkpoints.
Images, annotations, and weights use **CC BY-NC 4.0**: attribution and
noncommercial use. The source code's MIT license does not cover these assets.

## 1. Get the tools

Clone this repository and run commands from its root in an isolated Python 3.9
environment. Dataset download and inspection do not need a GPU or PyTorch.

```shell
git clone https://github.com/yongyong98/CurConMix-TMI.git
cd CurConMix-TMI
python -m pip install -e .
curconmix-assets --list
```

See [installation](installation.md) for environment activation and the historical
model runtime. Installing the package alone downloads no data or weights.

## 2. Download and install LLS48

```shell
curconmix-assets --asset lls48-dataset --destination data/LLS48 --install
python -m curconmix_release.lls48 verify --root data/LLS48
```

The first command downloads 25 independent image ZIPs and one metadata ZIP,
checks every archive's size and SHA-256, and installs their contents safely.
The second reads every selected image to verify per-file SHA-256, the unchanged
annotation, original frame order, five-frame clip boundaries, and the folds.
There are **9,560 images, 1,912 independent clips, and 48 procedures**.

The dataset download is about 26.42 GB (24.61 GiB). Allow **55 GiB of free space**
for download archives, installed images, and temporary extraction. The downloader
retains its cache; by default it is `data/.curconmix-cache/lls48-dataset`.
For the dataset plus all three MRTT checkpoint variants, allow **75 GiB** including
their retained caches. Use `--cache-dir` to put the cache on a different disk:

```shell
curconmix-assets --asset lls48-dataset --destination data/LLS48 --cache-dir downloads/lls48 --install
```

The [separate spatial release](spatial_checkpoints.md) adds LLS48 B224/B384 and
CholecT45 L224 bundles. Allow 90 GiB in total if retaining all of those as well
as the dataset, MRTT weights, and download caches; download only the variants
needed for your experiment.

Use a filesystem that supports hard links, such as NTFS, ext4, or APFS, for both
cache and installation. The installer checks this before starting downloads;
FAT/exFAT destinations are not supported by its atomic no-overwrite installation.

If interrupted, rerun the same command. Completed, checksum-verified parts are
reused; the unfinished part is downloaded again. This is part-level resume, not
HTTP byte-range resume. Existing installed files are reused only when identical;
different files are preserved and cause an error. Do not merge another dataset
version into this destination.

Once installed, this check needs no network access, including when the cache has
been removed by the user:

```shell
curconmix-assets --asset lls48-dataset --destination data/LLS48 --install --verify-only
```

After successful verification, you may remove only the relevant download-cache
directory to reclaim space. Keep the installed dataset and repository manifest.
The inline file checksums still permit offline verification without the cache;
a later normal install command will download missing cache parts again.

See the [dataset card](dataset_card.md) for the schema, class dictionaries,
license, and interpretation. Manual downloads are also available on the release
page: extract **all 26 ZIPs into the same LLS48 directory**, then run the verifier.
Do not concatenate the ZIP files; each one is a valid, independent archive.
For manual integrity checks, the release also provides
[`lls48_sha256sums.txt`](https://github.com/yongyong98/CurConMix-TMI/releases/download/lls48-v1.0.0/lls48_sha256sums.txt)
with the exact SHA-256 of all 41 dataset/checkpoint files. The installer performs
these checks automatically.

## 3. Read frames or independent clips

The lightweight index preserves canonical row order and procedure-level folds:

```python
from curconmix_release.dataset import LLS48Index

frames = LLS48Index("data/LLS48")
sample = frames[0]
print(sample["image_path"], sample["video"], sample["clip_name"])
print(len(sample["labels"]))  # 266: 195 triplet + 17 I + 16 V + 38 T

training = LLS48Index("data/LLS48", fold=0, split="train")
validation = LLS48Index("data/LLS48", fold=0, split="val")
for clip in validation.clips():
    assert len(clip) == 5
    # Load the five original image paths with your chosen image library.
    # Keep this clip independent when extracting segmental-evaluation events.
    break
```

`train` means the other four procedure folds; `val` means the held-out fold.
There is no additional test split implied by these names. The index does not
decode, resize, normalize, augment, or generate features. It is not a claim that
arbitrary preprocessing reproduces the paper. The original PNGs have two image
sizes, so a raw-image batch cannot assume a single shape without preprocessing.

## 4. Download an MRTT variant

Each asset installs all five fold-specific weights. Choose the variant matching
your qualified spatial features; these are **temporal weights, not image encoders**.

| Asset ID | Feature dimension | Download size, five folds | Config |
| --- | ---: | ---: | --- |
| `lls48-mrtt-b224` | 1,024 | 1.79 GB | `lls48_base224_plus.json` |
| `lls48-mrtt-b384` | 1,024 | 1.79 GB | `lls48_base384_plus.json` |
| `lls48-mrtt-l224` | 1,536 | 4.32 GB | `lls48_large224_plus.json` |

```shell
curconmix-assets --asset lls48-mrtt-b224 --destination artifacts/checkpoints --install
curconmix-assets --asset lls48-mrtt-b224 --destination artifacts/checkpoints --install --verify-only
```

The resulting directory is `artifacts/checkpoints/lls48/b224/`, containing
`best_model_fold0.pth` through `best_model_fold4.pth`. Substitute `b384` or `l224`
in the asset ID to install another variant. All 15 weights total 7.89 GB.
Checkpoint caches are also retained; plan for both the cache and installed files.

With the [documented PyTorch runtime](installation.md#mrtt-model-execution)
installed, check a weight file and strict-load its model without images, features,
training, or inference:

```shell
python -m curconmix_release.checkpoint --asset lls48-mrtt-b224 --root artifacts/checkpoints --fold 0
```

This verifies the file hash before loading, checks fold and architecture metadata,
and prints a short JSON result. It constructs the model on CPU but never calls
its forward method. Allow several GiB of available RAM, particularly for L224.
`--help` works without PyTorch.

## 5. What can run now, and what still needs qualified inputs?

You can download and inspect the dataset, use its unchanged annotations and
folds, and safely load the released MRTT state dictionaries. Model-only exports
were checked against the originals tensor-by-tensor and strict-loaded into the
public MRTT implementation; no new predictions were generated during release.

**This is not yet a numerically qualified raw-image-to-prediction pipeline.**
Matching B224/B384 spatial students are available in the
[separate spatial release](https://github.com/yongyong98/CurConMix-TMI/releases/tag/spatial-encoders-v1.0.0).
A new opt-in exporter preserves frame keys and emits labels and a checked feature
manifest; follow the [training and evaluation workflow](reproduction.md).
Its preprocessing and output contracts have software tests, not a new comparison
against historical features or published scores. Stored feature candidates were not released because exact
image-to-feature-row provenance could not be established. Matching shapes and
GT alone do not prove image identity. See [model availability](model_zoo.md).

If you already have independently qualified, matching features and labels,
follow the layout in the [README](../README.md#prepare-feature-based-execution).
Then validate and preview inference without running it:

```shell
curconmix preflight --config configs/temporal/lls48_base224_plus.json --feature-root data/features --label-root data/labels --inference-only
curconmix infer-temporal --config configs/temporal/lls48_base224_plus.json --feature-root data/features --label-root data/labels --checkpoint-dir artifacts/checkpoints/lls48/b224 --output outputs/lls48_b224.csv --dry-run
```

Remove `--dry-run` only when intentionally running inference with the documented
runtime and qualified inputs. Dataset installation never launches training or
inference. Downloading a checkpoint does not by itself reproduce a paper score.

For LLS48, segmental F1 is **clip-bounded class-wise multi-label segmental F1**.
Reset event extraction for each five-frame clip. Do not concatenate clips or
interpolate across the approximately 90-second unsampled gaps. This measures
local temporal consistency, not dense transition localization or continuity
across gaps. The model may consume procedure-ordered observed features; that is
distinct from the segmental evaluation boundary. Inference with a validated
`--feature-manifest` restores original keys after full-GT checks. Without that
option, its synthetic IDs must not be used directly for clip-bounded F1.
