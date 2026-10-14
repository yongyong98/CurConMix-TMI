# Third-party notices

Third-party notices apply independently of the CurConMix authors' code license.
The authors' own software is licensed under the MIT License in `LICENSE`; this
does not replace third-party licenses. LLS48 and checkpoint release terms are
separately selected as CC BY-NC 4.0; see DATA_AND_MODEL_LICENSE.md.

## ivtmetrics

The dataset-specific `ivtmetrics*` modules under `src/curconmix_repro/temporal/`
and `legacy/spatial/` are based on the CAMMA action-triplet metric implementation,
including project-specific mappings and modifications. Existing source headers
are preserved. Upstream: <https://github.com/CAMMA-public/ivtmetrics>.

Copyright (c) 2022, Research Group CAMMA. BSD 2-Clause License; the full notice is
retained in [third_party/licenses/ivtmetrics.txt](third_party/licenses/ivtmetrics.txt).

## SupContrast

The supervised contrastive loss implementations in `legacy/spatial/*/losses.py`
and their weighted adaptations derive from SupContrast. Upstream:
<https://github.com/HobbitLong/SupContrast>.

Copyright (c) 2020, Yonglong Tian. BSD 2-Clause License; the full notice is retained
in [third_party/licenses/SupContrast.txt](third_party/licenses/SupContrast.txt).

## Dependencies and datasets

PyTorch, torchvision, timm, and other dependencies retain their respective
licenses. CholecT45 must be obtained under the dataset provider's terms.
No clinical images, annotations, weights, or features are included in this code
checkout. Future artifacts must identify the terms applicable to each asset.
