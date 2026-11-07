# 两个 API 的真实模型测试报告

本次使用 120 条预先设计并标注的合成样本，通过实际 HTTP 顺序调用两个 API。
**整体 ID 准确率：85.83%（103/120）。**

结果只描述这批合成样本上的表现，不等同于公司真实项目数据上的准确率。
标签判断的是问题描述是否属于给定历史模式，没有源码上下文来证明当前问题确实为误判。

## 测试配置

- 模型：`AlexWortega/openjev/qwen3.5-4b-nli-v5`。
- 权重版本：`5df9876e03397de71504a8f156509e6529c263d8`。
- 运行环境：`{'python': '3.15.2', 'torch': '2.15.0+cpu', 'device': 'cpu', 'dtype': 'torch.float32', 'backend': 'TransformersNLI'}`；CPU 线程数 `6`。
- 阈值 `0.85`，矛盾概率上限 `0.1`，模式 `directional`。
- 关键词：`any`，区分大小写 `True`。
- 原有两条历史记录保持不变；每条样本先调用召回 API，再原样传入返回的候选数组调用匹配 API。
- 模型从本地加载，使用真实 Transformers 分类头及 Waitress HTTP 服务。阈值未按本测试结果调整。
- 数据集 SHA256：`692fd29fbbe12ba368190fada1b773e228e018ec6ea4aa086afb93f3b201f93a`。
- 开始（UTC）：`2026-11-05T23:13:57.225643+00:00`；结束：`2026-11-05T23:24:48.371846+00:00`。

## 指标

| 指标 | 结果 | 分母与含义 |
| --- | --- | --- |
| API1 候选集合准确率 | 100.00% | 120/120，严格按既定关键词规则比较完整候选集合 |
| API1 正例召回率 | 80.00% | 48/60 个正例被召回 |
| API2 非空候选准确率 | 94.90% | 93/98；真正执行模型比较的样本 |
| 两步整体 ID 准确率 | 85.83% | 103/120；ID 或 null 与标签完全相同 |
| 是否匹配的二分类准确率 | 85.83% | 103/120；不要求匹配 ID 正确 |
| 匹配 ID 精确率 | 100.00% | 返回的非空 ID 中，有多少 ID 正确 |
| 匹配 ID 召回率 | 71.67% | 全部正例中，有多少返回了正确 ID |
| 真实问题被错误过滤率 | 0.00% | 0/60 个负例返回了非空 ID |
| 错误历史 ID | 0 | 应匹配且返回了 ID，但不是正确的历史 ID |
| API 错误 | 0 | 错误不作为正确的 null 结果 |

二分类混淆矩阵：TP=43，FP=0，TN=60，FN=17。
TP/FP 的正类指“应当匹配历史误判”，API 错误单独计数，不混入 TN。

## 各类样本

| 类型 | 数量 | ID 正确数 | 准确率 |
| --- | ---: | ---: | ---: |
| 编译类正例（有关键词） | 24 | 24 | 100.00% |
| NOT_INIT 正例（有关键词） | 24 | 19 | 79.17% |
| 编译类同义表达（缺关键词） | 6 | 0 | 0.00% |
| 编译类构建失败表达（缺关键词） | 6 | 0 | 0.00% |
| 明确否定编译失败 | 10 | 10 | 100.00% |
| 含编译关键词的其他风险 | 10 | 10 | 100.00% |
| 明确否定 NOT_INIT 未定义 | 10 | 10 | 100.00% |
| NOT_INIT 已定义但存在其他风险 | 10 | 10 | 100.00% |
| 两个关键词都出现的负例 | 10 | 10 | 100.00% |
| 无关键词的其他风险 | 10 | 10 | 100.00% |

12 条正例故意采用不含“编译”的同义表达，用于展示关键词召回损失。
因此候选集合完全符合字符串规则，也不代表业务正例能全部进入模型阶段。

## 错例

| case_id | 类型 | 描述 | 预期 | 实际 | 候选 |
| --- | --- | --- | --- | --- | --- |
| case_029 | NOT_INIT 正例（有关键词） | NOT_INIT 并未在任何头文件中定义。 | no_init_issue | null | no_init_issue |
| case_032 | NOT_INIT 正例（有关键词） | NOT_INIT 的定义已经删除，这里仍然继续引用它。 | no_init_issue | null | no_init_issue |
| case_035 | NOT_INIT 正例（有关键词） | NOT_INIT 没有声明也没有定义，不能直接使用。 | no_init_issue | null | no_init_issue |
| case_037 | NOT_INIT 正例（有关键词） | NOT_INIT 只在其他配置中定义，当前配置下未定义。 | no_init_issue | null | no_init_issue |
| case_040 | NOT_INIT 正例（有关键词） | 没有任何有效的宏定义能够提供 NOT_INIT。 | no_init_issue | null | no_init_issue |
| case_049 | 编译类同义表达（缺关键词） | 构建阶段因语法错误无法生成目标文件。 | compile_issue | null | [] |
| case_050 | 编译类同义表达（缺关键词） | 源文件存在类型冲突，无法生成可执行程序。 | compile_issue | null | [] |
| case_051 | 编译类同义表达（缺关键词） | 构建失败：函数调用引用了未声明的类型。 | compile_issue | null | [] |
| case_052 | 编译类同义表达（缺关键词） | 代码中的模板参数错误使整个构建中断。 | compile_issue | null | [] |
| case_053 | 编译类同义表达（缺关键词） | 语法分析器报告缺少分号，目标文件无法生成。 | compile_issue | null | [] |
| case_054 | 编译类同义表达（缺关键词） | 类型检查无法通过，当前源程序不能转换为目标文件。 | compile_issue | null | [] |
| case_055 | 编译类构建失败表达（缺关键词） | 源程序缺少右括号，语法检查失败，无法生成目标文件。 | compile_issue | null | [] |
| case_056 | 编译类构建失败表达（缺关键词） | 此处引用的类根本没有声明，构建过程因此报错并停止。 | compile_issue | null | [] |
| case_057 | 编译类构建失败表达（缺关键词） | 构建期间发现重复的函数定义，目标程序无法生成。 | compile_issue | null | [] |
| case_058 | 编译类构建失败表达（缺关键词） | 类型检查器发现返回值不符合声明，因此源文件构建失败。 | compile_issue | null | [] |
| case_059 | 编译类构建失败表达（缺关键词） | 模板实参类型不满足约束，当前源程序构建不成功。 | compile_issue | null | [] |
| case_060 | 编译类构建失败表达（缺关键词） | 源代码中的非法语法导致构建失败，可执行文件无法生成。 | compile_issue | null | [] |

## 耗时与复现

计分请求阶段总耗时 421.58 秒，包含两步 HTTP 调用和结果写入；不包含模型下载、加载和预热。
所有成功样本平均两步请求耗时 3.51 秒，P95 为 9.06 秒。空候选跳过模型，平均值不是纯推理时延。

- 测试数据：`data\benchmark.v1.jsonl`。
- 每条 HTTP 响应：`reports/benchmark.v1.py315.complete/results.jsonl`。
- 机器可读统计与运行配置：`reports/benchmark.v1.py315.complete/summary.json`。

```powershell
$env:MODEL_PATH=(Resolve-Path '.\models\qwen3.5-4b-nli-v5').Path
$env:LOCAL_FILES_ONLY='true'; $env:HF_HUB_OFFLINE='1'
$env:DEVICE='cpu'; $env:DTYPE='float32'; $env:QUANTIZATION='none'
$env:BATCH_SIZE='1'; $env:THRESHOLD='0.85'; $env:MAX_CONTRADICTION='0.10'
$env:MATCH_MODE='directional'
python -m scripts.benchmark_api --serve-local --threads 6 --output reports/benchmark.v1.py315.complete
python -m scripts.report_benchmark reports/benchmark.v1.py315.complete --output BENCHMARK-PY315.md
```

建议下一步使用真实问题及人工复核标签建立独立测试集。新增关键词、细化历史描述或调整阈值后，另设运行目录比较；保留本次原始结果作为基线。
