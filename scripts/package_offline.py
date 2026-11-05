"""Export a self-contained Docker image and an offline deployment tar bundle."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "issue-filter-offline:20261105-py315-cpu"
MODEL_SHA256 = "2b5f6e0691a5ed0c5d0f5f3e49bb26abe2d8efd7c14fda4431cd803c12b78f68"
COMPOSE_SHA256 = "7758048ad38e580f22559603312874fec8bb638d021a9e64182385fc17cc12a2"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def docker(*arguments):
    return subprocess.run(["docker", *map(str, arguments)], check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def file_info(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "offline")
    parser.add_argument("--skip-save", action="store_true", help="Reuse an already exported image with matching ID")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    bundle = args.output_root / "issue-filter-offline-linux-amd64-cpu-py315-20261105"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "images").mkdir(exist_ok=True)
    (bundle / "host-tools").mkdir(exist_ok=True)
    (bundle / "data").mkdir(exist_ok=True)

    inspection = json.loads(docker("image", "inspect", args.image))[0]
    if inspection["Os"] != "linux" or inspection["Architecture"] != "amd64":
        raise ValueError("This bundle requires a Linux amd64 image")
    model = ROOT / "models/qwen3.5-4b-nli-v5"
    print("Checking local model and official Compose hashes...", flush=True)
    model_files = {path.name: file_info(path) for path in sorted(model.iterdir()) if path.is_file()}
    if model_files["model.safetensors"]["sha256"] != MODEL_SHA256:
        raise ValueError("Local model weights do not match the checkpoint SHA256")
    host_tools = args.output_root / "host-tools"
    if sha256(host_tools / "docker-compose-linux-x86_64") != COMPOSE_SHA256:
        raise ValueError("Compose binary does not match official release checksum")
    for name in ("docker-29.9.3.tgz", "docker-compose-linux-x86_64"):
        shutil.copy2(host_tools / name, bundle / "host-tools" / name)
    for path in (ROOT / "deploy/offline").iterdir():
        if path.is_file():
            shutil.copy2(path, bundle / path.name)
    shutil.copy2(ROOT / "data/historical_issues.json", bundle / "data/historical_issues.json")
    for name in ("README.md", "BENCHMARK.md", "BENCHMARK-PY315.md"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, bundle / name)

    image_tar = bundle / "images/issue-filter-offline-20261105-py315-cpu.tar"
    manifest_path = bundle / "manifest.json"
    if args.skip_save:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["image"]["id"] != inspection["Id"] or not image_tar.is_file():
            raise ValueError("Existing archive does not match the current image; export again")
    else:
        print("Exporting complete image with model (all parent layers included)...", flush=True)
        subprocess.run(["docker", "save", "--output", str(image_tar), args.image], check=True)

    print("Recording installed dependencies without network access...", flush=True)
    packages = docker("run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "python",
                      args.image, "-m", "pip", "freeze")
    (bundle / "python-packages.txt").write_text(packages, encoding="utf-8")
    docker("run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "python",
           args.image, "-m", "pip", "check")

    verification_path = args.output_root / "verification-py315.json"
    if not verification_path.is_file():
        raise ValueError("Missing offline/verification-py315.json; validate the actual image before packaging")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification["image_id"] != inspection["Id"]:
        raise ValueError("Verification belongs to a different image")
    if not verification.get("success"):
        raise ValueError("Offline verification did not pass; refusing to label this bundle ready")
    shutil.copy2(verification_path, bundle / "verification.json")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": "linux/amd64", "variant": "cpu-float32",
        "python_version": docker("run", "--rm", "--pull", "never", "--network", "none", "--entrypoint",
                                 "python", args.image, "--version").strip(),
        "image": {"tag": args.image, "id": inspection["Id"], "unpacked_bytes": inspection["Size"],
                  "layers": inspection["RootFS"]["Layers"]},
        "model": {"repository": "AlexWortega/openjev", "subfolder": "qwen3.5-4b-nli-v5",
                  "revision": "5df9876e03397de71504a8f156509e6529c263d8", "files": model_files,
                  "image_path": "/models/qwen3.5-4b-nli-v5"},
        "host_tools": {"docker": "29.9.3", "compose": "v5.7.0", "platform": "linux/amd64"},
        "sources": ["https://gallery.ecr.aws/docker/library/python",
                    "https://download.docker.com/linux/static/stable/x86_64/docker-29.9.3.tgz",
                    "https://github.com/docker/compose/releases/tag/v5.7.0",
                    "https://huggingface.co/AlexWortega/openjev"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Hashing all delivery files...", flush=True)
    hashes = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256(path)}  {path.relative_to(bundle).as_posix()}\n")
    (bundle / "SHA256SUMS").write_text("".join(hashes), encoding="ascii")

    archive = args.output_root / (bundle.name + ".tar")
    print(f"Creating complete transfer bundle: {archive}", flush=True)

    def permissions(info):
        if info.isdir():
            info.mode = 0o755
        elif info.name.endswith(".sh") or info.name.endswith("docker-compose-linux-x86_64"):
            info.mode = 0o755
        else:
            info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as output:
        output.add(bundle, arcname=bundle.name, filter=permissions)
    archive_hash = sha256(archive)
    (archive.parent / (archive.name + ".sha256")).write_text(f"{archive_hash}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"bundle": str(archive), "bytes": archive.stat().st_size,
                      "sha256": archive_hash, "image_id": inspection["Id"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
