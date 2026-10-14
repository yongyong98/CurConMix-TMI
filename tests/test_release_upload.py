"""No-network tests for the release upload integrity gate."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "upload_release_assets", ROOT / "tools/upload_release_assets.py"
)
UPLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPLOAD)


def test_server_digest_required():
    record = {"size_bytes": 7, "sha256": "a" * 64}
    remote = {
        "name": "file.zip",
        "state": "uploaded",
        "size": 7,
        "digest": "sha256:" + "a" * 64,
    }
    UPLOAD.validate_remote(record, remote)
    for changed in (
        {"size": 8},
        {"digest": None},
        {"state": "starter"},
        {"digest": "sha256:" + "b" * 64},
    ):
        with pytest.raises(ValueError):
            UPLOAD.validate_remote(record, {**remote, **changed})
