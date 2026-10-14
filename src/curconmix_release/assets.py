"""Verified asset downloads and non-overwriting multipart installation.

Downloads resume at verified file/part boundaries, not HTTP byte offsets. ZIPs
are staged one part at a time. Cached parts are retained for offline verification.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener
import zipfile


MAX_EXPANDED_SIZE = 100 * 1024**3
MAX_MEMBERS = 200000
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


class HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen(url, timeout):
    return build_opener(HTTPSOnlyRedirect()).open(url, timeout=timeout)


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") not in (1, 2) or not isinstance(
        manifest.get("assets"), list
    ):
        raise ValueError("Unsupported asset manifest")
    ids = [asset["id"] for asset in manifest["assets"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate asset IDs")
    for asset_id in ids:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", asset_id):
            raise ValueError("Invalid asset ID")
    return manifest


def _validate_url(url):
    parsed = urlparse(url or "")
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("A public HTTPS download URL without credentials is required")


def _relative_path(value, allow_root=False):
    if allow_root and value == ".":
        return Path()
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(
        part in ("", ".", "..")
        or part.endswith((".", " "))
        or any(ord(char) < 32 for char in part)
        or any(char in '<>"|?*' for char in part)
        or part.split(".", 1)[0].upper() in WINDOWS_RESERVED
        for part in parts
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*parts)


def _validate_hash_size(item):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", item.get("sha256") or ""):
        raise ValueError("A verified SHA-256 digest is required")
    if type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0:
        raise ValueError("A verified positive file size is required")


def asset_files(asset):
    """Return v2 files, or a legacy single-file entry without changing its API."""
    return asset["files"] if "files" in asset else [asset]


def validate_downloadable(asset):
    if asset.get("status") != "published" or asset.get("available") is not True:
        raise ValueError(
            f"Asset {asset['id']} is not published; no download is available."
        )
    if asset.get("license_status") != "approved":
        raise ValueError("Asset release terms have not been approved")
    files = asset_files(asset)
    if not isinstance(files, list) or not files:
        raise ValueError("A nonempty file list is required")
    names = set()
    for item in files:
        _validate_url(item.get("url"))
        _validate_hash_size(item)
        if "files" not in asset:
            continue
        filename = _relative_path(item.get("filename"))
        if len(filename.parts) != 1 or filename.name.casefold() in names:
            raise ValueError("Part filenames must be unique safe basenames")
        names.add(filename.name.casefold())
        if item.get("format") not in ("zip", "file"):
            raise ValueError("File format must be zip or file")
        _relative_path(
            item.get(
                "install_path", "." if item["format"] == "zip" else item["filename"]
            ),
            allow_root=item["format"] == "zip",
        )
        if "expanded_size_bytes" in item and (
            type(item["expanded_size_bytes"]) is not int
            or item["expanded_size_bytes"] <= 0
            or item["expanded_size_bytes"] > MAX_EXPANDED_SIZE
        ):
            raise ValueError("Invalid ZIP expanded_size_bytes")


def _assert_plain_path(path):
    """Reject symlinks and Windows junctions in every existing ancestor."""
    path = Path(os.path.abspath(path))
    for entry in (path, *path.parents):
        try:
            metadata = entry.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(metadata.st_mode)
            or getattr(metadata, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError(f"Symlink or reparse-point destination rejected: {entry}")


def _probe_hardlink_support(directory):
    """Fail before downloading when atomic no-overwrite publication is unsupported.

    Probe the nearest existing directory without creating the future dataset or
    cache root. Only the two uniquely named probe files are ever removed.
    """
    directory = Path(os.path.abspath(directory))
    _assert_plain_path(directory)
    while not directory.exists():
        directory = directory.parent
    if not directory.is_dir():
        raise ValueError(f"Destination ancestor is not a directory: {directory}")
    source = None
    link = None
    linked = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=".curconmix-hardlink-probe-", delete=False
        ) as handle:
            source = Path(handle.name)
            handle.write(b"1")
        link = source.with_name(source.name + ".link")
        os.link(source, link)
        linked = True
    except OSError as exc:
        raise OSError(
            f"Asset downloads/installations require a writable filesystem with "
            f"hard-link support (for example NTFS, ext4, or APFS): {directory}. "
            f"Choose another cache/destination directory. No download was started."
        ) from exc
    finally:
        if linked:
            link.unlink(missing_ok=True)
        if source is not None:
            source.unlink(missing_ok=True)


def _digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def verify_file(item, path):
    path = Path(path)
    _assert_plain_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    if _digest_file(path) != (item["size_bytes"], item["sha256"].lower()):
        raise ValueError(f"Existing file size or SHA-256 mismatch; preserved: {path}")
    return path


def download_asset(asset, destination):
    """Legacy single-file API: always refuse an existing destination."""
    validate_downloadable(asset)
    if "files" in asset:
        raise ValueError("Use process_asset for multipart assets")
    return _download_file(asset, destination)


def _download_file(item, destination):
    destination = Path(destination)
    _assert_plain_path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    _probe_hardlink_support(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        size = 0
        with urlopen(item["url"], timeout=60) as response:
            _validate_url(response.geturl())
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".download-", delete=False
            ) as handle:
                temporary = Path(handle.name)
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > item["size_bytes"]:
                        raise ValueError("Downloaded file exceeds the registered size")
                    digest.update(block)
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        if size != item["size_bytes"] or digest.hexdigest() != item["sha256"].lower():
            raise ValueError(
                "Downloaded file size or SHA-256 does not match the manifest"
            )
        # Same-directory hard-link publication is atomic and refuses overwrite.
        # If the filesystem cannot support it, fail without publishing a partial file.
        _assert_plain_path(destination)
        os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _zip_members(archive, item):
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ValueError("ZIP has too many members")
    total = 0
    seen = {}
    spellings = {}
    files = []
    for info in infos:
        if info.orig_filename != info.filename:
            raise ValueError("Unsafe normalized ZIP filename")
        name = info.filename[:-1] if info.is_dir() else info.filename
        relative = _relative_path(name)
        key = relative.as_posix().casefold()
        for length in range(1, len(relative.parts) + 1):
            spelling = "/".join(relative.parts[:length])
            previous = spellings.setdefault(spelling.casefold(), spelling)
            if previous != spelling:
                raise ValueError("Case-colliding ZIP path")
        if key in seen:
            raise ValueError("Duplicate or case-colliding ZIP member")
        seen[key] = info.is_dir()
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("ZIP symlink or special file rejected")
        if info.flag_bits & 1:
            raise ValueError("Encrypted ZIP member rejected")
        if not info.is_dir():
            if stat.S_IFMT(mode) == stat.S_IFDIR:
                raise ValueError("ZIP directory metadata is inconsistent")
            total += info.file_size
            files.append((info, relative))
    for key in seen:
        for ancestor in PurePosixPath(key).parents:
            if seen.get(str(ancestor)) is False:
                raise ValueError("ZIP file/directory collision")
    limit = item.get("expanded_size_bytes", MAX_EXPANDED_SIZE)
    if total > limit or total > MAX_EXPANDED_SIZE:
        raise ValueError("ZIP exceeds expanded size limit")
    if "expanded_size_bytes" in item and total != item["expanded_size_bytes"]:
        raise ValueError("ZIP expanded size does not match the manifest")
    if "member_count" in item and len(files) != item["member_count"]:
        raise ValueError("ZIP member count does not match the manifest")
    if not files:
        raise ValueError("Empty ZIP archive rejected")
    return files


def _prefix(item):
    return _relative_path(
        item.get("install_path", "." if item["format"] == "zip" else item["filename"]),
        allow_root=item["format"] == "zip",
    )


def _zip_records(item, cached, staging=None, only_paths=None):
    """Stream all members, checking CRC and optional member digests before publish."""
    records = []
    expected = None
    if "members" in item:
        expected = {}
        for member in item["members"]:
            name = _relative_path(member["path"]).as_posix()
            if name in expected:
                raise ValueError("Duplicate expected ZIP member")
            _validate_hash_size(member)
            expected[name] = member
    with zipfile.ZipFile(cached) as archive:
        for info, relative in _zip_members(archive, item):
            target = (
                staging / relative
                if staging is not None
                and (only_paths is None or relative.as_posix() in only_paths)
                else None
            )
            output = None
            try:
                if target is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = target.open("xb")
                digest = hashlib.sha256()
                size = 0
                with archive.open(info) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        size += len(block)
                        if size > info.file_size:
                            raise ValueError("ZIP member exceeds its declared size")
                        digest.update(block)
                        if output is not None:
                            output.write(block)
                if size != info.file_size:
                    raise ValueError("ZIP member size mismatch")
                record = {
                    "path": relative.as_posix(),
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                }
                if expected is not None:
                    reference = expected.pop(record["path"], None)
                    if reference is None or (size, record["sha256"]) != (
                        reference["size_bytes"],
                        reference["sha256"].lower(),
                    ):
                        raise ValueError("ZIP member SHA-256 or membership mismatch")
                records.append(record)
            finally:
                if output is not None:
                    output.close()
    if expected:
        raise ValueError("Expected ZIP members are missing")
    return records


def _publish_file(source, target, record):
    _assert_plain_path(target)
    if target.exists():
        return verify_file(record, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_plain_path(target)
    try:
        os.link(source, target)
    except FileExistsError:
        verify_file(record, target)
    return target


def _install_zip(item, cached, destination, verify_only):
    prefix = _prefix(item)
    if verify_only:
        if cached.is_file():
            verify_file(item, cached)
            records = _zip_records(item, cached)
        elif item.get("members"):
            records = item["members"]
        else:
            raise FileNotFoundError(
                "Offline ZIP verification requires the retained cached part"
            )
        for record in records:
            verify_file(record, destination / prefix / _relative_path(record["path"]))
        return len(records)
    missing = set()
    required_bytes = 0
    with zipfile.ZipFile(cached) as archive:
        for info, relative in _zip_members(archive, item):
            target = destination / prefix / relative
            _assert_plain_path(target)
            if not target.exists():
                missing.add(relative.as_posix())
                required_bytes += info.file_size
    if not missing:
        records = _zip_records(item, cached)
        for record in records:
            verify_file(record, destination / prefix / _relative_path(record["path"]))
        return len(records)
    _assert_plain_path(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required_bytes:
        raise OSError(
            f"Insufficient free space to stage this ZIP part ({required_bytes} bytes)"
        )
    # Same-volume staging: published hard links do not duplicate expanded bytes.
    with tempfile.TemporaryDirectory(
        prefix=".curconmix-install-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        records = _zip_records(item, cached, staging, missing)
        # Preflight every collision before publishing any file from this part.
        for record in records:
            target = destination / prefix / _relative_path(record["path"])
            _assert_plain_path(target)
            if target.exists():
                verify_file(record, target)
        for record in records:
            relative = _relative_path(record["path"])
            _publish_file(staging / relative, destination / prefix / relative, record)
    return len(records)


def process_asset(asset, destination, install=False, verify_only=False, cache_dir=None):
    """Download, install, or verify one manifest asset without replacing user files."""
    validate_downloadable(asset)
    destination = Path(destination)
    _assert_plain_path(destination)
    if "files" not in asset:
        if install:
            raise ValueError("Installation requires a schema-v2 files manifest")
        if verify_only or destination.exists():
            return [verify_file(asset, destination)]
        return [download_asset(asset, destination)]
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", asset["id"]):
        raise ValueError("Invalid asset ID")
    cache = (
        Path(cache_dir)
        if cache_dir is not None
        else (
            destination.parent / ".curconmix-cache" / asset["id"]
            if install
            else destination
        )
    )
    _assert_plain_path(cache)
    if destination.exists() and not destination.is_dir():
        raise ValueError("Multipart destination must be a directory")
    if install and not verify_only:
        # ZIP extraction stages beside the destination; raw files stage inside
        # it. Validate both locations before any potentially large downloads.
        _probe_hardlink_support(destination.parent)
        _probe_hardlink_support(destination)
    results = []
    for item in asset["files"]:
        cached = cache / item["filename"]
        if verify_only and install and item["format"] == "file":
            target = destination / _prefix(item)
            results.append(verify_file(item, target))
            continue
        if not (verify_only and install):
            if cached.exists() or verify_only:
                verify_file(item, cached)
                print(f"Verified cached part: {item['filename']}")
            else:
                print(
                    f"Downloading: {item['filename']} ({item['size_bytes']} bytes)",
                    flush=True,
                )
                _download_file(item, cached)
                print(f"Downloaded and verified: {item['filename']}")
        if install:
            if item["format"] == "zip":
                count = _install_zip(item, cached, destination, verify_only)
                print(
                    f"{'Verified' if verify_only else 'Installed'} {count} files: {item['filename']}"
                )
            else:
                target = destination / _prefix(item)
                # Cross-volume caches need a same-volume staging copy.
                _assert_plain_path(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    verify_file(item, target)
                else:
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            dir=target.parent, prefix=".install-", delete=False
                        ) as handle:
                            temporary = Path(handle.name)
                            with cached.open("rb") as source:
                                shutil.copyfileobj(source, handle, length=1024 * 1024)
                        verify_file(item, temporary)
                        _publish_file(temporary, target, item)
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
            results.append(destination)
        else:
            results.append(cached)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/assets.json"))
    parser.add_argument(
        "--list",
        action="store_true",
        help="List asset availability without network access",
    )
    parser.add_argument("--asset", help="Asset ID from the manifest")
    parser.add_argument(
        "--destination", type=Path, help="V2 cache/install directory; v1 output file"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install verified files/ZIP parts into destination",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify cached or installed files without network access",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Install cache (default: destination parent/.curconmix-cache/asset-id)",
    )
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.list:
            for asset in manifest["assets"]:
                detail = (
                    f"{len(asset_files(asset))} files"
                    if asset.get("available") is True
                    else "no download"
                )
                print(f"{asset['id']}: {asset['status']} ({detail})")
            return 0
        if not args.asset or not args.destination:
            parser.error("provide --list, or both --asset and --destination")
        selected = [asset for asset in manifest["assets"] if asset["id"] == args.asset]
        if len(selected) != 1:
            raise ValueError(f"Unknown asset ID: {args.asset}")
        if args.cache_dir and not args.install:
            parser.error("--cache-dir requires --install")
        result = process_asset(
            selected[0],
            args.destination,
            args.install,
            args.verify_only,
            args.cache_dir,
        )
        print(
            f"Verified {'installation' if args.install else 'download'}: {result[-1]}"
        )
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
