"""Create a readable report from persisted HTTP responses without re-inference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.benchmark_api import summarize

CATEGORY_NAMES = {
    "compile_positive": "编译类正例（有关键词）",
    "not_init_positive": "NOT_INIT 正例（有关键词）",
    "compile_keyword_miss": "编译类同义表达（缺关键词）",
    "compile_keyword_miss_extra": "编译类构建失败表达（缺关键词）",
    "compile_negation": "明确否定编译失败",
    "compile_related_but_different": "含编译关键词的其他风险",
    "not_init_negation": "明确否定 NOT_INIT 未定义",
    "not_init_related_but_different": "NOT_INIT 已定义但存在其他风险",
    "two_candidates_negative": "两个关键词都出现的负例",
    "no_keyword_negative": "无关键词的其他风险",
}


def percent(value):
    return "不适用" if value is None else f"{value:.2%}"


def cell(value):
    return ("null" if value is None else str(value)).replace("|", "\\|").replace("\n", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="Directory containing summary.json and results.jsonl")
    parser.add_argument("--output", type=Path, default=Path("BENCHMARK.md"))
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    dataset_path = Path(summary["dataset"])
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != summary["dataset_sha256"]:
        raise ValueError("Dataset changed since the run; refusing to produce a misleading report")
    cases = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    if [row["case_id"] for row in rows] != [case["case_id"] for case in cases]:
        raise ValueError("Results do not cover the complete dataset in its original order")
    for row, case in zip(rows, cases, strict=True):
        if any(row[key] != case[key] for key in case):
            raise ValueError("Saved case or label differs from the fixed dataset")
    metrics = summarize(rows)
    summary["metrics"] = metrics
    summary["by_category"] = {category: summarize([row for row in rows if row["category"] == category])
                              for category in Counter(row["category"] for row in rows)}
    # Local serving explicitly sets preload_model=True before creating the app.
    if summary["local_server"]:
        summary["configuration"]["preload_model"] = True
    (args.run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config = summary["configuration"]
    matrix = metrics["confusion_matrix"]
    lines = [
        "# 两个 API 的真实模型测试报告", "",
        f"本次使用 {len(rows)} 条预先设计并标注的合成样本，通过实际 HTTP 顺序调用两个 API。",
        f"**整体 ID 准确率：{percent(metrics['exact_accuracy'])}（{metrics['exact_correct']}/{metrics['samples']}）。**", "",
        "结果只描述这批合成样本上的表现，不等同于公司真实项目数据上的准确率。",
        "标签判断的是问题描述是否属于给定历史模式，没有源码上下文来证明当前问题确实为误判。", "",
        "## 测试配置", "",
        f"- 模型：`{config['model_id']}/{config['model_subfolder']}`。",
        f"- 权重版本：`{config['model_revision']}`。",
        f"- 运行环境：`{summary.get('runtime', {})}`；CPU 线程数 `{summary.get('cpu_threads')}`。",
        f"- 阈值 `{config['threshold']}`，矛盾概率上限 `{config['max_contradiction']}`，模式 `{config['match_mode']}`。",
        f"- 关键词：`{config['keyword_mode']}`，区分大小写 `{config['keyword_case_sensitive']}`。",
        "- 原有两条历史记录保持不变；每条样本先调用召回 API，再原样传入返回的候选数组调用匹配 API。",
        "- 模型从本地加载，使用真实 Transformers 分类头及 Waitress HTTP 服务。阈值未按本测试结果调整。",
        f"- 数据集 SHA256：`{summary['dataset_sha256']}`。",
        f"- 开始（UTC）：`{summary['started_at_utc']}`；结束：`{summary['finished_at_utc']}`。", "",
        "## 指标", "",
        "| 指标 | 结果 | 分母与含义 |", "| --- | --- | --- |",
        f"| API1 候选集合准确率 | {percent(metrics['candidate_set_accuracy'])} | {metrics['candidate_set_correct']}/{len(rows)}，严格按既定关键词规则比较完整候选集合 |",
        f"| API1 正例召回率 | {percent(metrics['keyword_recall'])} | {metrics['keyword_recall_hits']}/{metrics['positive_samples']} 个正例被召回 |",
        f"| API2 非空候选准确率 | {percent(metrics['nonempty_candidate_accuracy'])} | {metrics['nonempty_candidate_exact_correct']}/{metrics['nonempty_candidate_samples']}；真正执行模型比较的样本 |",
        f"| 两步整体 ID 准确率 | {percent(metrics['exact_accuracy'])} | {metrics['exact_correct']}/{len(rows)}；ID 或 null 与标签完全相同 |",
        f"| 是否匹配的二分类准确率 | {percent(metrics['binary_accuracy'])} | {metrics['binary_correct']}/{len(rows)}；不要求匹配 ID 正确 |",
        f"| 匹配 ID 精确率 | {percent(metrics['matched_id_precision'])} | 返回的非空 ID 中，有多少 ID 正确 |",
        f"| 匹配 ID 召回率 | {percent(metrics['matched_id_recall'])} | 全部正例中，有多少返回了正确 ID |",
        f"| 真实问题被错误过滤率 | {percent(metrics['false_filter_rate'])} | {matrix['FP']}/{metrics['negative_samples']} 个负例返回了非空 ID |",
        f"| 错误历史 ID | {metrics['wrong_ids']} | 应匹配且返回了 ID，但不是正确的历史 ID |",
        f"| API 错误 | {metrics['api_errors']} | 错误不作为正确的 null 结果 |", "",
        f"二分类混淆矩阵：TP={matrix['TP']}，FP={matrix['FP']}，TN={matrix['TN']}，FN={matrix['FN']}。",
        "TP/FP 的正类指“应当匹配历史误判”，API 错误单独计数，不混入 TN。", "",
        "## 各类样本", "",
        "| 类型 | 数量 | ID 正确数 | 准确率 |", "| --- | ---: | ---: | ---: |",
    ]
    for category, values in summary["by_category"].items():
        lines.append(f"| {CATEGORY_NAMES.get(category, category)} | {values['samples']} | {values['exact_correct']} | {percent(values['exact_accuracy'])} |")
    lines += ["", "12 条正例故意采用不含“编译”的同义表达，用于展示关键词召回损失。",
              "因此候选集合完全符合字符串规则，也不代表业务正例能全部进入模型阶段。", "",
              "## 错例", "",
              "| case_id | 类型 | 描述 | 预期 | 实际 | 候选 |", "| --- | --- | --- | --- | --- | --- |"]
    errors = [row for row in rows if not row["success"] or row["expected_id"] != row["predicted_id"]]
    for row in errors:
        lines.append("| " + " | ".join(cell(value) for value in (
            row["case_id"], CATEGORY_NAMES.get(row["category"], row["category"]), row["description"],
            row["expected_id"], row["predicted_id"] if row["success"] else "API_ERROR", ", ".join(row["candidate_ids"]) or "[]",
        )) + " |")
    if not errors:
        lines.append("| — | — | 本次无错例 | — | — | — |")
    lines += ["", "## 耗时与复现", "",
              f"计分请求阶段总耗时 {summary['wall_seconds']:.2f} 秒，包含两步 HTTP 调用和结果写入；不包含模型下载、加载和预热。",
              f"所有成功样本平均两步请求耗时 {metrics['mean_seconds']:.2f} 秒，P95 为 {metrics['p95_seconds']:.2f} 秒。空候选跳过模型，平均值不是纯推理时延。", "",
              f"- 测试数据：`{summary['dataset']}`。",
              f"- 每条 HTTP 响应：`{args.run.as_posix()}/results.jsonl`。",
              f"- 机器可读统计与运行配置：`{args.run.as_posix()}/summary.json`。", "",
              "```powershell",
              "$env:MODEL_PATH=(Resolve-Path '.\\models\\qwen3.5-4b-nli-v5').Path",
              "$env:LOCAL_FILES_ONLY='true'; $env:HF_HUB_OFFLINE='1'",
              "$env:DEVICE='cpu'; $env:DTYPE='float32'; $env:QUANTIZATION='none'",
              "$env:BATCH_SIZE='1'; $env:THRESHOLD='0.85'; $env:MAX_CONTRADICTION='0.10'",
              "$env:MATCH_MODE='directional'",
              ".\\.venv\\Scripts\\python.exe -m scripts.benchmark_api --serve-local --threads 6 --output reports/benchmark.v1.fp32",
              ".\\.venv\\Scripts\\python.exe -m scripts.report_benchmark reports/benchmark.v1.fp32",
              "```", "",
              "建议下一步使用真实问题及人工复核标签建立独立测试集。新增关键词、细化历史描述或调整阈值后，另设运行目录比较；保留本次原始结果作为基线。", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
