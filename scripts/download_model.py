"""Download only the chosen checkpoint, not the whole multi-checkpoint repo."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from issue_filter.config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models", help="Parent directory for the checkpoint subfolder")
    args = parser.parse_args()
    settings = Settings.from_env()
    subfolder = settings.model_subfolder.strip("/")
    if not subfolder or ".." in Path(subfolder).parts:
        parser.error("MODEL_SUBFOLDER must be a nonempty checkpoint subfolder")
    snapshot_download(
        repo_id=settings.model_id,
        revision=settings.model_revision,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=args.output,
    )
    print(f"MODEL_PATH={Path(args.output).resolve() / subfolder}")


if __name__ == "__main__":
    main()
