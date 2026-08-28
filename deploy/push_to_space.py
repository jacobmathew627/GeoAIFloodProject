"""
Publish the dashboard to a Hugging Face Space.

Why this is a script and not `git push hf main`
------------------------------------------------
The Space has to contain things this repository deliberately does not track:
`display/`, `models/live_model.npz` and `outputs/conformal_sets.tif` are
derived artefacts, regenerable from ~5 GB of LFS rasters by the pipeline, and
committing them to GitHub would spend LFS quota on data one command
reproduces (see .gitignore).

Hugging Face builds the Space from *its own* repository, so those files have
to be present there. Rather than reversing the gitignore decision, this
assembles a self-contained Space repo in a temporary directory: tracked source
plus the ~33 MB of runtime artefacts, and nothing else. GitHub stays
source-only; the Space gets what it needs to build.

What lands in the Space
-----------------------
    Dockerfile              its final stage is `app`, which is what HF builds
    README.md               deploy/SPACE_README.md, whose YAML front-matter is
                            what tells HF this is sdk: docker on port 8501
    app.py, src/, static/, .streamlit/
    display/                21 MB of display-resolution layers
    models/                 live_model.npz, rainfall_forecast.joblib, *.json
    outputs/                conformal_sets.tif

Excluded: GeoAI_New/ (3.7 GB), data_aligned/ (1.3 GB), data/ (874 MB), tests/,
evaluation/, docs/, the archived .pth/.h5 weights, and the per-scenario hazard
rasters -- the API's, not the dashboard's.

Usage
-----
    pip install huggingface_hub
    python deploy/push_to_space.py --space <user>/<space-name> --token hf_...

The token needs *write* scope. Create one at
<https://huggingface.co/settings/tokens>. Pass it with --token or set
$HF_TOKEN; it is never written to disk by this script.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGGER = logging.getLogger("geoai_flood")

#: (source, destination) pairs copied into the Space, relative to PROJECT_ROOT.
#: Directories are copied wholesale; files individually.
PAYLOAD: Tuple[Tuple[str, str], ...] = (
    ("Dockerfile", "Dockerfile"),
    ("deploy/SPACE_README.md", "README.md"),
    ("app.py", "app.py"),
    ("requirements.txt", "requirements.txt"),
    ("src", "src"),
    ("static", "static"),
    (".streamlit", ".streamlit"),
    ("display", "display"),
    ("models/live_model.npz", "models/live_model.npz"),
    ("models/rainfall_forecast.joblib", "models/rainfall_forecast.joblib"),
    ("outputs/conformal_sets.tif", "outputs/conformal_sets.tif"),
)

#: Every models/*.json is small and at least one is read by the UI.
JSON_GLOB = "models/*.json"

#: Large binaries the Space needs, which HF tracks with LFS. Matching
#: .gitattributes to the payload keeps the push from inlining a 21 MB raster.
LFS_PATTERNS = ("*.tif", "*.npz", "*.joblib")

IGNORE_IN_SRC = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def assemble(target: Path) -> int:
    """Copy the payload into `target`. Returns total bytes written."""
    total = 0
    missing = []

    for src_rel, dst_rel in PAYLOAD:
        src = PROJECT_ROOT / src_rel
        dst = target / dst_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE_IN_SRC, dirs_exist_ok=True)
            total += sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        else:
            shutil.copy2(src, dst)
            total += dst.stat().st_size

    for src in sorted(PROJECT_ROOT.glob(JSON_GLOB)):
        dst = target / "models" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total += dst.stat().st_size

    if missing:
        raise FileNotFoundError(
            "Missing build artefacts: " + ", ".join(missing) + "\n\nBuild them first:\n"
            "  python src/make_display_rasters.py\n"
            "  python src/live_model.py --build\n"
            "  python src/rainfall_forecast.py --snapshot"
        )

    (target / ".gitattributes").write_text(
        "\n".join(f"{p} filter=lfs diff=lfs merge=lfs -text" for p in LFS_PATTERNS) + "\n",
        encoding="utf-8",
    )
    return total


def push(space: str, token: str, message: str, dry_run: bool = False) -> str:
    """Assemble and push. Returns the Space URL."""
    if "/" not in space:
        raise ValueError(f"--space must be '<user>/<name>', got {space!r}")

    with tempfile.TemporaryDirectory(prefix="hf-space-") as tmp:
        work = Path(tmp) / "space"
        work.mkdir()

        LOGGER.info("Assembling Space payload...")
        total = assemble(work)
        LOGGER.info(
            "  %.1f MB across %d files", total / 1e6, sum(1 for _ in work.rglob("*") if _.is_file())
        )

        if dry_run:
            LOGGER.info("Dry run: assembled but not pushed. Contents:")
            for f in sorted(work.rglob("*")):
                if f.is_file():
                    LOGGER.info("    %8.2f MB  %s", f.stat().st_size / 1e6, f.relative_to(work))
            return f"https://huggingface.co/spaces/{space}"

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        LOGGER.info("Ensuring Space %s exists...", space)
        api.create_repo(repo_id=space, repo_type="space", space_sdk="docker", exist_ok=True)

        LOGGER.info("Uploading (this pushes ~%.0f MB)...", total / 1e6)
        api.upload_folder(
            repo_id=space,
            repo_type="space",
            folder_path=str(work),
            commit_message=message,
        )

    url = f"https://huggingface.co/spaces/{space}"
    LOGGER.info("Pushed. The Space builds the Dockerfile's final stage (app).")
    LOGGER.info("  %s", url)
    return url


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Publish the dashboard to a Hugging Face Space")
    parser.add_argument("--space", help="Target Space as '<user>/<name>'")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token")
    parser.add_argument("--message", default="Deploy Ernakulam flood risk dashboard")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble and list the payload without creating or pushing anything",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.dry_run:
        if not args.space:
            parser.error("--space is required unless --dry-run")
        if not args.token:
            parser.error("No token. Pass --token or set $HF_TOKEN (needs write scope).")

    try:
        url = push(
            space=args.space or "<user>/<space>",
            token=args.token or "",
            message=args.message,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)
    print(url)


if __name__ == "__main__":  # pragma: no cover
    main()
