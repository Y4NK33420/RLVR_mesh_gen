from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def _resolve_token(env_path: Path | None) -> str:
    env_candidates = ["HUGGINGFACE_TOKEN", "HUGGING_FACE_TOKEN", "HF_TOKEN"]

    for key in env_candidates:
        value = os.environ.get(key)
        if value:
            return value

    if env_path is not None and env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key in env_candidates and value.strip():
                return value.strip()

    raise RuntimeError(
        "No Hugging Face token found. Set HUGGINGFACE_TOKEN or HUGGING_FACE_TOKEN in environment or .env."
    )


def _download_one(
    repo: str,
    token: str,
    local_dir: Path,
    path_in_repo: str,
    force_download: bool,
) -> str:
    return hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=path_in_repo,
        token=token,
        local_dir=local_dir.as_posix(),
        force_download=force_download,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download only files listed in a ShapeNet subset manifest")
    parser.add_argument(
        "--manifest",
        default="artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json",
        help="Subset manifest JSON path",
    )
    parser.add_argument("--env-file", default=".env", help="Path to env file containing Hugging Face token")
    parser.add_argument(
        "--output-dir",
        default="data/shapenet_subsets/furniture_plus_display_cap1000",
        help="Destination root directory",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download workers")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for test downloads; 0 means all")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if cached/local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo = str(payload["repo"])
    records = list(payload["records"])

    if args.limit > 0:
        records = records[: args.limit]

    token = _resolve_token(Path(args.env_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(records)
    print(f"repo: {repo}")
    print(f"files_to_download: {total}")
    print(f"output_dir: {output_dir.as_posix()}")

    completed = 0
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(
                _download_one,
                repo,
                token,
                output_dir,
                str(rec["path"]),
                bool(args.force_download),
            ): str(rec["path"])
            for rec in records
        }

        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
                completed += 1
                if completed % 100 == 0 or completed == total:
                    print(f"progress: {completed}/{total}")
            except Exception as exc:  # noqa: BLE001
                failures.append({"path": path, "error": str(exc)})

    print(f"completed: {completed}")
    print(f"failed: {len(failures)}")

    if failures:
        failure_path = output_dir / "download_failures.json"
        failure_path.write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
        print(f"failure_report: {failure_path.as_posix()}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
