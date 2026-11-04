"""Run inside the service container. Uses only stdlib and the local HTTP API."""

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

base = "http://127.0.0.1:8000"
headers = {"Content-Type": "application/json"}
if os.getenv("API_KEY"):
    headers["Authorization"] = "Bearer " + os.environ["API_KEY"]


def post(path, body):
    request = Request(base + path, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=600) as response:
        return json.load(response)


deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    try:
        with urlopen(base + "/readyz", timeout=3) as response:
            if response.status == 200:
                break
    except (HTTPError, URLError, TimeoutError):
        time.sleep(1)
else:
    raise RuntimeError("Model did not become ready in 600 seconds")

cases = [
    ("\u7f16\u8bd1\u4e0d\u901a\u8fc7", "compile_issue"),
    ("NOT_INIT \u672a\u5b9a\u4e49\uff0c\u5b58\u5728\u672a\u77e5\u6807\u8bc6\u7b26\u98ce\u9669\u3002", "no_init_issue"),
    ("\u7f16\u8bd1\u5df2\u7ecf\u901a\u8fc7\uff0c\u771f\u6b63\u7684\u95ee\u9898\u662f\u8fd0\u884c\u65f6\u7a7a\u6307\u9488\u8bbf\u95ee\u3002", None),
]
for description, expected in cases:
    candidates = post("/api/v1/potential-issues", {"description": description})
    result = post("/api/v1/match", {"description": description, "potential_historical_issue_list": candidates})
    if result != {"matched_issue_id": expected}:
        raise AssertionError({"description": description, "expected": expected, "actual": result})
print(json.dumps({"status": "passed", "cases": len(cases), "http_calls": len(cases) * 2,
                  "model_path": os.getenv("MODEL_PATH"), "hub_offline": os.getenv("HF_HUB_OFFLINE")}))
