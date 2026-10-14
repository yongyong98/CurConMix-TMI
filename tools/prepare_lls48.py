"""Maintainer-only packaging of approved, unchanged LLS48 source files."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curconmix_release.lls48 import prepare_dataset, sha256_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory outside the source"
    )
    args = parser.parse_args()
    report = prepare_dataset(
        args.source,
        args.output,
        ROOT / "legacy/spatial/lls48/preprocess.py",
        ROOT / "src/curconmix_repro/temporal/ivtmetrics_lls/maps.txt",
        ROOT / "docs/dataset_card.md",
        ROOT / "DATA_AND_MODEL_LICENSE.md",
    )
    private = ROOT / ".release-local/lls48-package-provenance.json"
    private.parent.mkdir(exist_ok=True)
    with private.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "source_root": str(args.source.resolve()),
                "output_root": str(args.output.resolve()),
                "source_annotation_sha256": report["summary"]["annotation_sha256"],
                "preprocess_sha256": sha256_file(
                    ROOT / "legacy/spatial/lls48/preprocess.py"
                ),
                "numeric_map_sha256": sha256_file(
                    ROOT / "src/curconmix_repro/temporal/ivtmetrics_lls/maps.txt"
                ),
                "release_manifest_sha256": sha256_file(
                    args.output / "release_files.json"
                ),
            },
            handle,
            indent=2,
        )
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "archives": len(report["files"]),
                "archive_bytes": sum(f["size_bytes"] for f in report["files"]),
                "validation": report["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
