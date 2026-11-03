"""List or download only ISL500 MediaPipe files for SignSpeak's current labels.

By default this performs a dry listing. Pass --download to fetch the matched
HDF5 files. The full ISL500 repository is very large; this avoids downloading
videos, pose arrays, and unrelated words.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import time

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "ISL500/ISL-DATA"
MEDIA_PIPE_ROOT = "Landmarks/MediaPipe"
LABELS = {
    "actor",
    "adult",
    "afternoon",
    "alive",
    "alright",
    "animal",
    "bad",
    "bag",
    "ball",
    "bank",
    "bathroom",
    "beautiful",
    "doctor",
}
DEFAULT_OUTPUT = Path.home() / "Downloads" / "ISL500-SignSpeak"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"where to save matched .h5 files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the listed files; without this flag, only list them",
    )
    args = parser.parse_args()

    api = HfApi()
    print(f"Listing only {MEDIA_PIPE_ROOT}/ in {REPO_ID} …")
    entries = api.list_repo_tree(
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=MEDIA_PIPE_ROOT,
        recursive=True,
        expand=False,
    )

    matches: list[str] = []
    for entry in entries:
        path = getattr(entry, "path", "")
        name = PurePosixPath(path).name.lower()
        if name.endswith(".h5") and any(name.startswith(f"{label}_") for label in LABELS):
            matches.append(path)

    matches.sort()
    print(f"Found {len(matches)} matching landmark files for {len(LABELS)} labels.")
    if not matches:
        print("No matching files found. Check the dataset's current folder and filename layout.")
        return

    for path in matches:
        print(path)

    if not args.download:
        print("\nDry listing only. Re-run with --download to fetch these files.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading to {args.output_dir} …")

    def download(path: str) -> str:
        for attempt in range(1, 6):
            try:
                return hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=path,
                    local_dir=str(args.output_dir),
                )
            except Exception as error:
                if attempt == 5:
                    raise
                delay = min(2 ** (attempt - 1), 16)
                print(
                    f"Temporary error for {path}: {error}. "
                    f"Retrying in {delay}s ({attempt}/5)."
                )
                time.sleep(delay)

    completed = 0
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download, path): path for path in matches}
        for future in as_completed(futures):
            completed += 1
            try:
                local_path = future.result()
                print(f"[{completed}/{len(matches)}] {local_path}")
            except Exception as error:
                path = futures[future]
                failed.append((path, str(error)))
                print(f"[{completed}/{len(matches)}] FAILED {path}: {error}")

    if failed:
        print(f"\n{len(failed)} files still failed after retries.")
        print("Run the same --download command again to retry; completed files are kept locally.")
        raise SystemExit(1)
    print("Download complete. Review the dataset license before using the files.")


if __name__ == "__main__":
    main()
