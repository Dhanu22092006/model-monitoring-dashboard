"""Safe persistence of fitted pipelines.

Pickle/joblib files can execute code when loaded, so a model is only loaded when
(1) its path resolves inside the artifact directory and (2) its SHA-256 equals the
hash recorded in the database when *we* trained and saved it.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import joblib


class ArtifactError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_pipeline(pipeline, directory, filename):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    joblib.dump(pipeline, path, compress=3)
    return filename, sha256_file(path), path.stat().st_size


@lru_cache(maxsize=8)
def _load_verified(resolved_path: str, expected_sha: str):
    return joblib.load(resolved_path)


def load_pipeline(directory, filename, expected_sha256):
    base = Path(directory).resolve()
    path = (base / filename).resolve()
    if base not in path.parents:
        raise ArtifactError("Artifact path is outside the artifact directory.")
    if not path.is_file():
        raise ArtifactError("Model artifact file is missing.")
    if not expected_sha256 or sha256_file(path) != expected_sha256:
        raise ArtifactError("Model artifact failed its integrity check (SHA-256 mismatch); refusing to load.")
    return _load_verified(str(path), expected_sha256)
