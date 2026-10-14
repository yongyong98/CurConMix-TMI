# Data access and evaluation

## Availability

The authors confirm that institutional approval and de-identification of LLS48
have been completed. They have selected **CC BY-NC 4.0** for the LLS48 images and
annotations and for model checkpoints. See the
[data and model license notice](../DATA_AND_MODEL_LICENSE.md).

The [LLS48 release](https://github.com/yongyong98/CurConMix-TMI/releases/tag/lls48-v1.0.0)
contains curated images and unchanged annotations, outside the Git source
history. Follow the [download quickstart](quickstart_lls48.md) for automatic
installation, checksums, offline verification, and a frame/clip reader. The asset
manifest separates published assets from unpublished configurations; the
download command rejects unpublished assets.

Keep clinical data outside the source repository and use only data you are
authorized to access. Derived-feature distribution remains pending and is not
enabled by the checkpoint inventory or the data-download entry.

## LLS48 benchmark selection

The canonical evaluation selection contains 48 procedures and 1,912 independently
sampled clips, each with exactly five frames: 9,560 images in total. Only this
annotation-selected subset defines the benchmark. Additional files in a local
data collection are not automatically part of the benchmark.

The action annotation schema contains 195 triplet classes, 17 instrument classes,
16 verb classes, and 38 target classes. The dataset also has five step classes and
18 task classes. Class mappings, null-triplet handling, and procedure-level fold
assignments must be preserved when preparing an authorized local copy. All
observations from a procedure belong to the same cross-validation fold.

## Temporal interpretation

LLS48 is not a densely annotated continuous-video benchmark. The clips are
separated by approximately 90-second unsampled intervals. For **clip-bounded
class-wise multi-label segmental F1**, each five-frame clip is a separate temporal
unit. Event extraction resets at every clip boundary. Events must never cross a
clip boundary, and clips must not be concatenated or interpolated across gaps.

This metric measures local temporal consistency within annotated clips. It does
not measure dense transition localization, persistence through unsampled gaps,
or continuous-clock long-range reasoning. The reported comparison uses a
probability threshold of 0.5 and temporal IoU of 0.25.

Action AP follows the dataset's procedure-wise aggregation and existing
triplet-to-component decomposition. Procedure-wise AP aggregation does **not**
turn LLS48 segmental F1 into a procedure-level continuous-timeline evaluation.
Keep the two metric protocols distinct.

CholecT45 is evaluated over complete annotated video timelines. Obtain it through
the [official CholecT45 project](https://github.com/CAMMA-public/cholect45) and
follow its access terms; this repository does not redistribute it.

## Download package

The package includes only the canonical image selection, its byte-identical
annotation table, four class mappings, procedure-level folds, ordered frame/clip
metadata, a dataset card, CC BY-NC 4.0 terms, and file checksums. There are 25 image
ZIPs and one metadata ZIP, each below 1 GiB. All archives extract into the same
dataset root. The 925 additional PNGs outside the canonical annotation selection
are excluded. Public metadata contains pseudonymous IDs and relative paths, not
internal storage paths. See the [dataset card](dataset_card.md).

See [model availability](model_zoo.md) for the independent status of model weights.
