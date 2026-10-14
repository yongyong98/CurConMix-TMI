# Features from newly trained, inner-validation-selected students

This workflow is separate from the released-checkpoint exporter. It currently
supports **LLS48 SwinB(224), the public 195-triplet/266-output schema only**.
It does not support CholecT45, B384/L224 new-run export, or the unresolved AIS
193-class experiment. It is not a reconstruction of historical training.

The new student files must come from the explicit inner-validation spatial
training workflow, with five checkpoint `.provenance.json` sidecars. Checkpoint
SHA-256, embedded/sidecar metadata, dataset, backbone, fold, annotation hash,
training-split identity, and train/validation/test procedure lists must agree.
Released historical weights are not accepted as a new-run warm start here.

## Plan and model-free checks

Run from a source checkout. The default only prints a plan; absent data or
weights are not downloaded and no model libraries are imported.

```bash
python -m curconmix_release.training_feature_export --config configs/temporal/lls48_base224_plus.json --training-split /local/approved-training-split.json --dataset-root /local/lls48 --checkpoint-root /local/new-spatial-run/checkpoints --output /local/new-training-features
```

Add `--preflight` to check the canonical annotation, image-file existence,
explicit split and checkpoint sidecars/SHA-256. This reads files but performs
no image decoding, model loading, forward pass or output writing. No default
inner split is generated. The expected checkpoint basename is
`fold{fold}_swin_bas_266_CurConMix_B224_Student.pth`; an explicit
`--checkpoint-pattern` may change only this basename, not the required lineage.

## Explicit feature execution

Only after the new encoders exist and their protocol has been approved, adding
`--execute --device cuda` opts into spatial feature extraction. This performs
encoder forward passes, not training. It uses the existing validation
preprocessing and pre-head pooled backbone features, in the pinned spatial
environment (`requirements-spatial.txt`). It never silently downloads an
ImageNet initialization: restricted checkpoint loading constructs the model
with `pretrained=False` and strict state loading.

Output must be a new directory with an existing parent. Existing files are
never overwritten. A failed execution preserves its unpublished staging
directory for diagnosis. Successful output contains:

- `features/lls48/<configured-model-name>/<fold>/train/VIDxx.npy`: both training
  and inner-validation procedures; their separate membership remains explicit
  in `training_feature_manifest.json` and the approved split.
- `labels/lls48/triplet_label/VIDxx.npy`: unchanged ordered action labels.
- `training_feature_manifest.json`: schema 2, annotation row keys, feature/label
  hashes, source-code identity, new-student provenance, and explicit partitions.

The logical configured model name retains the public path convention; it is
not a claim that these arrays came from the released historical checkpoints.
The manifest contains absolute local annotation/checkpoint paths. Treat it as
a local experiment artifact, not a file to publish with private paths.

## Optional final-test arrays, never checkpoint selection

By default no outer-test feature array is written for each fold. If final
held-out inference will be needed, specify `--include-outer-test` when creating
the new export. The already-fixed, inner-validation-selected encoder then also
extracts those procedures into `<fold>/val/VIDxx.npy`, recorded under
`folds[fold].outer_test`. These arrays are never fitting or model-selection
inputs; the temporal training validator checks only train/inner-validation
arrays. Final inference must explicitly validate outer-test lineage.

Choose this option before training the temporal model: the model is bound to
the complete manifest hash. Do not append outer-test arrays or change the
manifest after training, and do not replace the manifest on a trained model to
bypass identity checks. An export with different contents is a different
artifact requiring its own provenance.

## Interpretation and verification limits

Original CSV row order and five-frame clip identities are retained. Procedure
arrays represent sparse sampled observations, not a continuous clock. LLS48
segmental evaluation remains clip-bounded and cannot bridge unsampled gaps.
Synthetic tests and metadata/array checks do not establish numerical equality
with historical features, successful model execution, or reproduced paper AP.
No actual training or feature inference was run while preparing this workflow.
