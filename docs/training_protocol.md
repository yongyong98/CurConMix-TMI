# Independent validation for new training runs

New spatial and MRTT training requires an explicit procedure-level split manifest.
Each published outer fold is a final test set. Its remaining procedures must be
partitioned into nonempty, disjoint `train` and `inner_validation` sets **before**
training. There is no automatic validation split or fallback to test scores.

The same manifest applies to contrastive caches, pretraining, teacher training,
teacher soft labels, student training, feature extraction, and MRTT training.
Optimizer updates and contrastive caches use `train` only. Teacher, student and
MRTT checkpoints are selected by inner-validation IVT AP. Contrastive pretraining
uses its fixed curriculum schedule, not outer-test AP. Epoch counts and any
hyperparameter search must be fixed or selected without consulting outer tests.

## Supply a manifest

Use a UTF-8 JSON object with these fields:

| Field | Required value |
| --- | --- |
| `schema_version` | integer `1` |
| `protocol` | `nested_procedure_validation_v1` |
| `dataset` | `lls48` or `cholect45` |
| `annotation_sha256` | SHA-256 of the exact input annotation CSV |
| `selection_metric` | `AP_IVT` |
| `folds` | keys `0`, `1`, `2`, `3`, `4` |

Each fold has exactly three explicit lists of original `VIDxx` identifiers:
`train`, `inner_validation`, and `outer_test`. Their union must be the complete
dataset; their intersections must be empty; `outer_test` must equal the published
membership of that fold. List order is recorded and reused. The library will
not invent these choices for an investigator.

An optional `patient_groups` object maps every procedure to an anonymized patient
ID and enforces patient-disjoint partitions. Without it, the check establishes
procedure separation only, not patient identity. Do not put identifying patient
information in a repository or public manifest.

Pass the manifest as `--training-split protocol/splits.json` to cache preparation
and spatial stages. Direct legacy training entry points also require
`training_split=/absolute/path/to/splits.json`; omitting it is an error.

## Feature lineage and temporal training

New student checkpoints include their partition/protocol metadata and a
`<checkpoint>.provenance.json` sidecar with a file checksum. Training features
must come from the corresponding fold's newly trained student, with matching
split and annotation identities. Released model-only weights remain available
for inference; they are not automatically certified for a newly declared inner
split. Do not manually add provenance to old weights or feature arrays.

The version-2 training feature manifest records the encoder identity, array
hashes, original frame keys, and annotation-aligned labels. Fit and inner-
validation arrays occupy the outer-development `train` directory. Final-test
features, if explicitly exported after encoder selection, occupy `val`; they
are never read by the temporal training dataset or its checkpoint selector.

```shell
curconmix train-temporal --config configs/temporal/lls48_base224_plus.json --training-split protocol/splits.json --training-feature-manifest data/new_features/training_feature_manifest.json --feature-root data/new_features/features --label-root data/new_features/labels --output outputs/new_inner_validation_run --dry-run
```

Remove `--dry-run` only to deliberately launch training in a compatible isolated
environment. Default help and dry runs do not train, load a model, or download
weights. New training summaries contain **inner-validation selection AP**, not
outer-test performance. Evaluate the frozen selected checkpoints on each outer
test only after all choices have been finalized. New-checkpoint inference also
requires the matching `--training-split` and `--training-feature-manifest`, with
explicit outer-test feature records.

Use the [new-student feature exporter](training_features.md) for the currently
supported LLS48 B224 route. Choose `--include-outer-test` at the initial export
if final inference will be needed: the temporal checkpoint records the entire
manifest hash, so adding entries after training is not permitted. Training still
opens only the train/inner-validation feature arrays.

## Reporting boundary

The downloadable datasets and released checkpoints are unchanged. New runs with
this protocol produce new experimental results; they must not be described as
the runs that generated the paper's reported numbers. Source/unit tests verify
partition routing and provenance checks, not numerical score reproduction.

This package contains action-level spatial/MRTT training. Separate high-level
TeCNO step/task training is not implemented here. LLS48 remains sparsely sampled:
do not infer continuous time coverage from ordered procedure features; segmental
action evaluation remains independently clip-bounded.
