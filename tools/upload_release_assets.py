"""Upload an explicit checked asset list to an existing GitHub draft release.

Maintainer-only. No release is created or published here. Existing assets must
match their server-side SHA-256 and size; they are never overwritten or deleted.
The input plan is local-only: {repository, tag, files: [{path, sha256, size_bytes}]}.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess


def gh(*arguments):
    result = subprocess.run(
        ["gh", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def validate_remote(record, remote):
    if remote.get("state") != "uploaded":
        raise ValueError(f"Remote asset is not complete: {remote.get('name')}")
    if remote.get("size") != record["size_bytes"]:
        raise ValueError(f"Remote size mismatch: {remote.get('name')}")
    if remote.get("digest") != f"sha256:{record['sha256']}":
        raise ValueError(f"Remote SHA-256 mismatch/missing: {remote.get('name')}")


def upload(plan, workers=2):
    repository, tag = plan["repository"], plan["tag"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("Invalid release tag")
    # The tag endpoint returns 404 for drafts; find the authenticated draft by
    # its exact tag in the paginated release list instead.
    pages = json.loads(
        gh("api", "--paginate", "--slurp", f"repos/{repository}/releases?per_page=100")
    )
    matches = [
        release for page in pages for release in page if release["tag_name"] == tag
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one existing release for this tag")
    release = matches[0]
    if not release.get("draft"):
        raise ValueError("Only an existing draft release may receive assets")
    existing = {asset["name"]: asset for asset in release["assets"]}
    files = plan["files"]
    names = [Path(item["path"]).name for item in files]
    if len(names) != len(set(names)) or not files:
        raise ValueError("Explicit unique asset basenames are required")
    for item in files:
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Local file size mismatch: {path.name}")
        if not 0 < item["size_bytes"] < 2 * 1024**3:
            raise ValueError("Each GitHub Release asset must be below 2 GiB")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("SHA-256 required")

    def one(item):
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError(f"Local SHA-256 mismatch: {path.name}")
        if path.name in existing:
            validate_remote(item, existing[path.name])
            print(f"Verified existing remote asset: {path.name}", flush=True)
            return existing[path.name]
        print(f"Uploading: {path.name} ({item['size_bytes']} bytes)", flush=True)
        gh("release", "upload", tag, str(path), "--repo", repository)
        current = json.loads(gh("api", f"repos/{repository}/releases/{release['id']}"))
        found = [a for a in current["assets"] if a["name"] == path.name]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one remote asset: {path.name}")
        validate_remote(item, found[0])
        print(f"Server SHA-256 and size verified: {path.name}", flush=True)
        return found[0]

    completed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, item) for item in files]):
            completed.append(future.result())
    return sorted(completed, key=lambda asset: asset["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("Existing upload receipt preserved")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    assets = upload(plan, args.workers)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(
            {"repository": plan["repository"], "tag": plan["tag"], "assets": assets},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
