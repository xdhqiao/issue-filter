"""Evaluate both real HTTP endpoints, persisting every response and failure.

--serve-local runs the actual Flask/Waitress app with TransformersNLI, never a
mock. Without it, --base-url targets an already running deployment.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Thread
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from issue_filter.config import Settings


def post_json(base_url, path, payload, timeout, api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(base_url + path, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers=headers, method="POST")
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response), perf_counter() - started
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(body)
        except ValueError:
            pass
        return error.code, body, perf_counter() - started
    except (URLError, TimeoutError, OSError) as error:
        return None, {"transport_error": str(error)}, perf_counter() - started


def summarize(rows):
    total = len(rows)
    successes = [row for row in rows if row["success"]]
    positives = sum(row["expected_id"] is not None for row in rows)
    negatives = total - positives
    exact = sum(row["success"] and row["predicted_id"] == row["expected_id"] for row in rows)
    predicted_positive = sum(row["success"] and row["predicted_id"] is not None for row in rows)
    correct_id = sum(row["success"] and row["predicted_id"] is not None
                     and row["predicted_id"] == row["expected_id"] for row in rows)
    tp = sum(row["success"] and row["predicted_id"] is not None and row["expected_id"] is not None for row in rows)
    fp = sum(row["success"] and row["predicted_id"] is not None and row["expected_id"] is None for row in rows)
    tn = sum(row["success"] and row["predicted_id"] is None and row["expected_id"] is None for row in rows)
    fn = sum(row["success"] and row["predicted_id"] is None and row["expected_id"] is not None for row in rows)
    keyword_hits = sum(row["recall_status"] == 200 and row["expected_id"] in row["candidate_ids"]
                       for row in rows if row["expected_id"] is not None)
    candidate_set_correct = sum(row["recall_status"] == 200 and
                                set(row["candidate_ids"]) == set(row["expected_candidate_ids"]) for row in rows)
    recalled_positives = [row for row in rows if row["expected_id"] is not None
                         and row["expected_id"] in row["candidate_ids"]]
    timings = sorted(row["recall_seconds"] + row["match_seconds"] for row in successes)
    return {
        "samples": total, "positive_samples": positives, "negative_samples": negatives,
        "successful_samples": len(successes), "api_errors": total - len(successes),
        "candidate_set_correct": candidate_set_correct,
        "candidate_set_accuracy": candidate_set_correct / total if total else None,
        "keyword_recall_hits": keyword_hits,
        "keyword_recall": keyword_hits / positives if positives else None,
        "exact_correct": exact, "exact_accuracy": exact / total if total else None,
        "binary_correct": tp + tn, "binary_accuracy": (tp + tn) / total if total else None,
        "matched_id_precision": correct_id / predicted_positive if predicted_positive else None,
        "matched_id_recall": correct_id / positives if positives else None,
        "recalled_positive_id_recall": sum(row["success"] and row["predicted_id"] == row["expected_id"]
                                           for row in recalled_positives) / len(recalled_positives)
        if recalled_positives else None,
        "false_filter_rate": fp / negatives if negatives else None,
        "wrong_ids": tp - correct_id,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "mean_seconds": sum(timings) / len(timings) if timings else None,
        "p95_seconds": timings[min(len(timings) - 1, int(.95 * len(timings)))] if timings else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmark.v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.v1"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--serve-local", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--threads", type=int, default=6, help="PyTorch CPU threads for local serving")
    parser.add_argument("--limit", type=int, default=0, help="Debug subset only; zero means full dataset")
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    cases = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if args.limit:
        cases = cases[:args.limit]
    if not cases or len({row["case_id"] for row in cases}) != len(cases):
        parser.error("dataset must be nonempty with unique case_id")
    settings = Settings.from_env()
    args.output.mkdir(parents=True, exist_ok=True)
    # Never include API_KEY in saved configuration.
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in asdict(settings).items() if key != "api_key"}
    metadata = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_sha256": hashlib.sha256(raw).hexdigest(), "dataset": str(args.dataset),
                "local_server": args.serve_local, "configuration": config,
                "cpu_threads": args.threads if args.serve_local else None}
    server = thread = None
    if args.serve_local:
        import torch
        from waitress import create_server
        from issue_filter import create_app

        torch.set_num_threads(args.threads)
        settings = replace(settings, preload_model=True)
        print("Loading real model and warming up local HTTP server...", flush=True)
        app = create_app(settings)
        backend = app.extensions["issue_matcher"].backend
        metadata["runtime"] = {"torch": torch.__version__, "device": backend.device,
                               "dtype": str(next(backend.model.parameters()).dtype),
                               "backend": type(backend).__name__}
        server = create_server(app, host="127.0.0.1", port=0, threads=2)
        thread = Thread(target=server.run, daemon=True)
        thread.start()
        args.base_url = f"http://127.0.0.1:{server.effective_port}"
    metadata["base_url"] = args.base_url
    results = []
    started = perf_counter()
    try:
        with (args.output / "results.jsonl").open("w", encoding="utf-8") as output:
            for index, case in enumerate(cases, 1):
                recall_status, candidates, recall_seconds = post_json(
                    args.base_url, "/api/v1/potential-issues", {"description": case["description"]},
                    args.timeout, settings.api_key,
                )
                candidate_ids = [item["id"] for item in candidates] if recall_status == 200 and isinstance(candidates, list) else []
                if recall_status == 200 and isinstance(candidates, list):
                    match_status, match_response, match_seconds = post_json(
                        args.base_url, "/api/v1/match", {"description": case["description"],
                        "potential_historical_issue_list": candidates}, args.timeout, settings.api_key,
                    )
                else:
                    match_status, match_response, match_seconds = None, None, 0
                success = match_status == 200 and isinstance(match_response, dict) and "matched_issue_id" in match_response
                row = {**case, "candidate_ids": candidate_ids,
                       "recall_status": recall_status, "match_status": match_status,
                       "recall_response": candidates, "match_response": match_response,
                       "recall_seconds": recall_seconds, "match_seconds": match_seconds,
                       "success": success,
                       "predicted_id": match_response["matched_issue_id"] if success else None}
                results.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(f"[{index}/{len(cases)}] {case['case_id']} expected={case['expected_id']} "
                      f"predicted={row['predicted_id']} status={match_status} "
                      f"seconds={recall_seconds + match_seconds:.2f}", flush=True)
                if not success:
                    print(f"API failure: {match_response or candidates}", flush=True)
    finally:
        if server is not None:
            server.task_dispatcher.shutdown()
            server.close()
            thread.join(timeout=2)
    summary = {**metadata, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": perf_counter() - started, "metrics": summarize(results),
               "by_category": {category: summarize([row for row in results if row["category"] == category])
                               for category in Counter(row["category"] for row in results)}}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved responses and metrics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
