"""Start the actual bundled image without networking and verify its APIs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def run(*arguments, **kwargs):
    return subprocess.run(["docker", *map(str, arguments)], check=True, capture_output=True,
                          text=True, encoding="utf-8", **kwargs).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="issue-filter-offline:20261104-cpu")
    parser.add_argument("--output", type=Path, default=Path("offline/verification.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    inspection = json.loads(run("image", "inspect", args.image))[0]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "image": args.image,
              "image_id": inspection["Id"], "platform": f"{inspection['Os']}/{inspection['Architecture']}",
              "network_mode": "none", "mounts": [], "success": False}
    container = run("run", "-d", "--pull", "never", "--network", "none", args.image).strip()
    try:
        print(f"Checking real offline container {container[:12]}...", flush=True)
        run("exec", container, "python", "-m", "pip", "check")
        report["dependency_check"] = "passed"
        manifest_code = (
            "import os,json,pathlib; p=pathlib.Path(os.environ['MODEL_PATH']); "
            "print(json.dumps({'path':str(p),'files':{f.name:f.stat().st_size for f in p.iterdir() if f.is_file()},"
            "'hub_offline':os.environ.get('HF_HUB_OFFLINE'),'local_files_only':os.environ.get('LOCAL_FILES_ONLY')}))"
        )
        report["model_files"] = json.loads(run("exec", container, "python", "-c", manifest_code))
        if report["model_files"]["files"].get("model.safetensors") != 9078635984:
            raise RuntimeError("Bundled weights are missing or incomplete")
        # check_api.py waits for the model and invokes both actual HTTP endpoints.
        check_code = (root / "deploy/offline/check_api.py").read_text(encoding="utf-8")
        process = subprocess.Popen(["docker", "exec", "-i", container, "python", "-"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8")
        process.stdin.write(check_code)
        process.stdin.close()
        deadline = time.monotonic() + 660
        while process.poll() is None:
            state = json.loads(run("inspect", container))[0]["State"]
            if not state["Running"]:
                process.terminate()
                report["container_state"] = state
                raise RuntimeError(f"Container exited: OOMKilled={state.get('OOMKilled')} ExitCode={state['ExitCode']}")
            if time.monotonic() > deadline:
                process.terminate()
                raise TimeoutError("Offline model HTTP check timed out")
            time.sleep(3)
        stdout, stderr = process.stdout.read(), process.stderr.read()
        if process.returncode:
            report["api_check_stderr"] = stderr
            raise RuntimeError(f"API self-check failed: {stderr}")
        report["api_check"] = json.loads(stdout.strip().splitlines()[-1])
        actual = json.loads(run("inspect", container))[0]
        report["network_mode"] = actual["HostConfig"]["NetworkMode"]
        report["mounts"] = actual["Mounts"]
        if report["network_mode"] != "none" or report["mounts"]:
            raise RuntimeError("Verification must not rely on networking or mounted model files")
        report["success"] = True
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Gunicorn normally writes logs to stderr; record both streams.
        logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True, encoding="utf-8")
        report["container_logs"] = logs.stdout + logs.stderr
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        subprocess.run(["docker", "rm", "-f", container], check=True, capture_output=True)


if __name__ == "__main__":
    main()
