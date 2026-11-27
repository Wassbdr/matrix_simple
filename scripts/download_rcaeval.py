"""Download the pinned public RCAEval release outside the repository."""

import argparse
import hashlib
import re
import shutil
import tarfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ARCHIVE_URL = "https://ndownloader.figshare.com/files/71725531"
ARCHIVE_HASHES = {
    "sha256": "ab41dbc65701395363246be92c4a9355bb27460a6895f6834d907812fd295404",
    "md5": "53bcd2afeec448b9bc7d8ffe9c3f9fd4",
}
HF_REPO = "phamquiluan/RCAEval"
HF_REVISION = "8b6c06a7a15bcecdbf70b611a13b789c5925dc17"
HF_INDEX_HASH = "ec0008c90d0c97076bc98370ac824c3c5a49be95d00bd6d6044ea8061861d271"
CASE = re.compile(r"re[123](ob|ss|tt)_[a-z0-9-]+_[a-z0-9]+_[0-9]+")
COUNTS = {"re1": 375, "re2": 270, "re3": 90}
TELEMETRY = {"metrics.json", "inject_time.txt", "logs.csv", "traces.csv", "root_cause.txt"}


def verify(path: Path, expected: dict[str, str]) -> None:
    hashes = {method: hashlib.new(method) for method in expected}
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            for digest in hashes.values():
                digest.update(chunk)
    for method, digest in hashes.items():
        if digest.hexdigest() != expected[method]:
            raise ValueError(f"{method} mismatch for {path}; remove the file and retry")


def complete(root: Path, suites: list[str], metrics: str) -> bool:
    return all(
        sum(
            (path.parent / "inject_time.txt").is_file()
            for path in root.glob(f"{suite}*/{metrics}")
        ) == COUNTS[suite]
        for suite in suites
    )


def figshare(root: Path, suites: list[str]) -> Path:
    out = root / "figshare" / "data"
    if complete(out, suites, "metrics.json"):
        return out
    archive = root / "RCAEval-v2.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with urlopen(ARCHIVE_URL, timeout=120) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, 2**20)
        verify(partial, ARCHIVE_HASHES)
        partial.replace(archive)
    else:
        verify(archive, ARCHIVE_HASHES)
    with tarfile.open(archive, mode="r|gz") as bundle:
        for item in bundle:
            parts = item.name.split("/")
            if len(parts) != 3 or parts[0] != "data":
                continue
            _, case, filename = parts
            if not (case[:3] in suites and CASE.fullmatch(case) and filename in TELEMETRY):
                continue
            if not item.isfile():
                raise ValueError(f"Unexpected archive entry: {item.name}")
            destination = out / case / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(".partial")
            source = bundle.extractfile(item)
            if source is None:
                raise ValueError(f"Cannot extract {item.name}")
            with source, partial.open("wb") as stream:
                shutil.copyfileobj(source, stream, 2**20)
            partial.replace(destination)
    if not complete(out, suites, "metrics.json"):
        raise ValueError(f"Incomplete RCAEval archive extraction at {out}")
    return out


def huggingface(root: Path, suites: list[str]) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface-hub and pyarrow to use the Hugging Face mirror.") from exc
    out = root / "hf"
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=HF_REVISION,
        allow_patterns=["cases.parquet", *(f"{suite}*/*" for suite in suites)],
        local_dir=out,
    )
    verify(out / "cases.parquet", {"sha256": HF_INDEX_HASH})
    if not complete(out, suites, "metrics.parquet"):
        raise ValueError(f"Incomplete RCAEval mirror at {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "data" / "rcaeval")
    parser.add_argument("--suites", nargs="+", choices=sorted(COUNTS), default=list(COUNTS))
    parser.add_argument("--source", choices=("auto", "figshare", "hf"), default="auto")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.source == "hf":
        print(huggingface(args.root, args.suites))
        return
    try:
        print(figshare(args.root, args.suites))
    except (URLError, TimeoutError, ConnectionError) as exc:
        if args.source != "auto":
            parser.exit(1, f"Figshare download failed: {exc}\n")
        print(f"Figshare download failed ({exc}); trying the Hugging Face mirror.")
        print(huggingface(args.root, args.suites))


if __name__ == "__main__":
    main()
