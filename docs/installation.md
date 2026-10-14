# Installation

Use a separate environment. Python 3.9 and PyTorch 2.0.1 are the historical
temporal reproduction stack. Newer versions are not claimed to reproduce the
published results. Run commands from the repository root.

## AP evaluation and data checks

AP evaluation does not require a GPU or PyTorch. With an activated Python 3.9
environment:

```shell
python -m pip install -r requirements-temporal.txt
python -m pip install -e ".[dev]"
curconmix --help
curconmix-assets --list
python -m pytest -q
```

To create and activate a new environment:

```shell
python -m venv .venv
```

On Linux/macOS, run `source .venv/bin/activate`. On Windows PowerShell, run
`.venv\Scripts\Activate.ps1` according to your organization's execution policy.

## MRTT model execution

Install PyTorch separately before explicitly running a model command. For the
historical CUDA 11.8 wheel on compatible hardware:

```shell
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only model inspection, substitute the CPU index:

```shell
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```

These commands follow the [official previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).
They pin a historical research runtime, not a current security-support guarantee.
Do not load untrusted checkpoints. The historical format uses Python pickle via
`torch.load`; the packaged temporal entry point requests `weights_only=True` and
fails if a checkpoint cannot be loaded safely. It does not silently retry with
unsafe unpickling. A checksum confirms file integrity, not trustworthiness.

## Spatial research implementation

See [spatial.md](spatial.md) for separate requirements, historical stages, and
remaining data/cache dependencies. Installing the temporal package alone does
not install the image preprocessing dependencies. The opt-in
[LLS48 feature and training workflow](reproduction.md) uses the spatial profile;
it has software tests but is not a measured end-to-end reproduction.
No data, pretrained weights, or feature caches are downloaded during installation
or tests. The separate CI spatial-environment job installs CPU dependencies,
checks imports and Hydra configuration, and runs synthetic tests without model
execution. That is an environment check, not a GPU training qualification.
