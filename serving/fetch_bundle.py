"""Download a registered model bundle from MLflow into a local directory.

Run as the serve pod's initContainer (using the train image, which has
mlflow): it resolves ``models:/<name>@<alias>`` to the logged bundle
artifact and stages it where the serve container expects ``MODEL_BUNDLE``.

    python -m serving.fetch_bundle \
        --model sta-sockshop --alias champion --dest /bundle

Requires MLFLOW_TRACKING_URI (and auth env if the server uses basic-auth).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch(model: str, alias: str, dest: str, artifact_path: str = "bundle") -> str:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise SystemExit("MLFLOW_TRACKING_URI is not set; cannot fetch the bundle.")
    import mlflow

    uri = f"models:/{model}@{alias}"
    logger.info("Resolving %s", uri)
    # download_artifacts on a models:/ URI fetches the whole logged artifact
    # tree; the bundle was logged under artifact_path="bundle".
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"{uri}/{artifact_path}")

    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    shutil.copytree(local, dest_path)

    meta = dest_path / "metadata.json"
    if not meta.exists():
        raise SystemExit(f"Fetched artifact has no metadata.json under {dest_path}")
    logger.info("Bundle staged at %s", dest_path)
    return str(dest_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sta-sockshop")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--dest", required=True, help="Target dir (= MODEL_BUNDLE)")
    parser.add_argument("--artifact-path", default="bundle")
    args = parser.parse_args(argv)
    fetch(args.model, args.alias, args.dest, args.artifact_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
