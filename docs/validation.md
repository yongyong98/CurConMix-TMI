# Validation performed

## Independent-validation training interfaces (2026-10-14)

Synthetic tests check disjoint outer-test/inner-validation/train membership,
encoder-stage and temporal-loader routing, selection metadata, artifact hashes,
annotation row/GT correspondence, and rejection of incompatible feature/decoder
lineage. New-run checkpoint selection uses inner validation only; the SSL stage
uses its fixed training schedule. The new keyed training-feature exporter is
currently LLS48 B224-only. No real model training, feature extraction, inference,
or numerical performance reproduction was executed for these interface changes.
See [the protocol](training_protocol.md) and [feature workflow](training_features.md).

The release-preparation checks below describe their original, bounded scope;
they do not certify a newly trained model or the new inner-split inputs.

## Published asset preparation

Checks performed during release preparation on 2026-10-12:

| Check | Outcome | What this does not establish |
| --- | --- | --- |
| CPU tests with synthetic fixtures | Current suite covers download, training serialization, order/fold checks, keyed features and cache preflight; see latest CI for the exact result | Published model performance |
| Python syntax compilation | Passed for packaged source | Runtime compatibility of every training dependency |
| Ruff error checks | Passed for `src`, `legacy`, `tools`, and `tests` | Complete style or scientific review |
| Staged whitespace check | Passed | Full supply-chain certification |
| Editable package installation | Passed; both command entry points work | Clean-machine dependency installation |
| Wheel build and license contents | Passed; MIT, BSD notices, and asset terms included | Standalone wheel data or model availability |
| Fresh Linux CPU installation and tests | Passed on GitHub Actions, Ubuntu 24.04 and Python 3.9 | GPU runtime or legacy spatial training dependencies |
| Real validation feature/label preflight | All six MRTT configurations passed | Image/row identity or checkpoint identity |
| Spatial stage previews and Hydra composition | Passed | Successful training or inference |
| New spatial input preflight | Real LLS48 annotation and 9,560 image paths checked; missing ten cache files rejected before model import | Decoding the images or executing training |
| New MRTT checkpoint serialization | Whole synthetic training payload restricted-loads; NumPy scalar regression covered, explicit fold enforced | A newly trained model or paper score |
| Procedure order replay | Captures observed order and validates/replays it without sorting | Recovery of unrecorded historical order or bitwise GPU determinism |
| Feature export and key restoration | Synthetic/mocked tests cover all folds, hashes, full GT, original keys, no-overwrite and preview-only behavior | Real spatial forward pass or numerical agreement with historical features |
| Contrastive cache preparation | Source-derived B224 recipe, explicit compute/download opt-in, input/resource/provenance checks | Execution of the cache generator or bit-identical historical caches |
| LLS48 package membership | Exactly 9,560 images, 1,912 five-frame clips, 48 procedures; 925 extras excluded | Independent clinical privacy certification |
| LLS48 package integrity | All 26 ZIP CRCs, exact membership, archive and per-file SHA-256 passed | New performance measurement |
| Annotation, taxonomy, and folds | Byte-identical master CSV and four dictionaries; 195 mappings agree with evaluator; canonical folds preserved | Relabeling or correction of class names |
| MRTT model-only exports | All 15 original/export tensor keys, shapes, dtypes, values identical; original hashes unchanged | Checkpoint-to-prediction numerical reproduction |
| Restricted checkpoint loading | All 15 load and strict-match public model in PyTorch 2.0.1 and 2.8, without forward passes | Runtime or numerical behavior of a full inference sequence |
| Spatial model-only exports | All 15 qualified exports passed tensor equality, stored-prediction pairing and strict loading in the variant-specific runtime | New image-based prediction reproduction |
| End-user checkpoint inspection | B224/B384/L224 fold 0 verified with hashes and strict CPU loading; help works without PyTorch | A forward pass or image-based inference |
| Metadata ZIP installation smoke test | Real 11-file metadata archive installs, resumes, and verifies offline with/without cache | A fresh full 26 GB dataset download and extraction |
| Published release inventories | Exact 42 dataset/MRTT assets and 16 spatial assets; all server SHA-256 digests, sizes and uploaded states match the allowlists through the anonymous public API | A full download of every remote asset |
| Anonymous public metadata round trip | Fresh 9,696,101-byte metadata ZIP download, SHA/CRC/exact 11-member checks, install, repeat install and offline checks with and without cache passed | A fresh full image-dataset download and extraction |
| Anonymous public spatial round trip | LLS48 B224 fold 0: all 349,785,025 bytes downloaded, SHA-256 matched, installed and restricted/strict CPU-loaded with PyTorch 2.0.1 and timm 0.5.4; 364 tensors | A network round trip of every checkpoint or any forward pass |

Both asset releases became public on 2026-10-12. The anonymous network checks
above cover the metadata ZIP and one spatial fold only. All 26 dataset ZIPs,
15 MRTT weights and 15 spatial weights were hashed locally; all 58 remote assets
(including the two checksum files) matched the exact server inventories. These
checks do not constitute a full fresh-machine data download or a reproduction
of the paper's predictions.

The CPU test environment used Python 3.9.13, NumPy 1.26.4, pandas 1.4.4, and
scikit-learn 1.5.1. This test environment is not the historical experiment lock.
Editable installation was checked in a separate virtual environment sharing
existing dependencies; the build backend was installed using normal build
isolation. A separate [GitHub Actions run](https://github.com/yongyong98/CurConMix-TMI/actions/runs/38192981121)
installed the CPU requirements and package in a fresh Ubuntu 24.04 environment
and passed its then-current 57 tests. Subsequent runs expand the suite. The new
spatial-environment CI job installs the CPU spatial dependency profile, checks
imports and all six stage configurations, and exercises restricted serialization
with PyTorch. Its result must be read from the matching commit's run; it does not
validate a GPU training run.

The real-input preflight covered the validation split of all five folds for each
configuration. Every CholecT45 configuration had 45 videos and 90,489 frames;
every LLS48 configuration had 48 procedures and 9,560 frames. It checked file
sets, folds, dimensions, float32 finite features, and binary labels. It did not
compare original image identities or execute a model.

The new tests cover multipart download/install, interrupted-part reuse, wrong
hashes, collisions, unsafe paths, symlinks, ZIP expansion limits, checkpoint export
metadata, release-manifest derivation, and canonical dataset structure. Tests use
synthetic fixtures, not clinical images or trained weights.

The usability changes also test the newly added workflow with synthetic arrays
and mocked model/image calls. No training, optimizer step or forward pass is
needed for those tests. Black checks cover new helper/test modules, portable CLI,
validation code and stage wrappers; inherited model/training files retain their
existing formatting to keep scientific changes reviewable.

Stored feature candidates passed full-GT, shape, fold, and finite-value checks,
but exact image-to-feature row identity was not established. Those legacy
features remain excluded. The new keyed exporter generates new features only
after an explicit user execution request; its software contract does not prove
equivalence to those historical arrays. Spatial release scope is separately
documented in [the checkpoint guide](spatial_checkpoints.md).

No training, optimizer steps, checkpoint inference, or performance reproduction
was performed during this preparation. Check the
[workflow page](https://github.com/yongyong98/CurConMix-TMI/actions/workflows/cpu-tests.yml)
for the latest commit's CPU validation status.
