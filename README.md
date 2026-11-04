# 代码审核误判问题过滤服务

Python 3.11 + Flask，将审核工具产生的**问题描述**与本地历史误判库比较：先按关键词召回候选，再使用本地 Qwen3.5-4B-NLI-v5 判断语义关系，返回达到阈值的最佳历史问题 ID；没有匹配时返回 `null`。本服务不负责调用上游审核大模型，也不重新读取代码仓库。

提供两个业务 API：

| API | 输入 | 输出 |
| --- | --- | --- |
| `POST /api/v1/potential-issues` | `description` 字符串 | 历史问题对象数组 |
| `POST /api/v1/match` | `description` 和 `potential_historical_issue_list` | `{"matched_issue_id": "..."}` 或 `{"matched_issue_id": null}` |

## 1. 模型选择与判断方式

核对日期：2026-11-04。采用社区模型仓库 [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) 中的 `qwen3.5-4b-nli-v5` 子目录；它基于 Qwen3.5 微调，并非 Qwen 官方发布的同名 NLI 服务。固定权重 revision 为 `5df9876e03397de71504a8f156509e6529c263d8`，避免上游更新导致行为漂移。作者当前仍推荐 v5；本项目不会自动追随“最新版”。

模型的架构是 `Qwen3_5ForSequenceClassification`，输出 `contradiction`（矛盾）、`entailment`（蕴含）、`neutral`（中立）三个概率。这里使用 Transformers 的 `AutoModelForSequenceClassification`，不使用聊天模板或 `generate()`。输入使用模型配置的 `nli_template`，默认格式为：

```text
Premise: 新审核问题描述
Hypothesis: 历史误判问题描述
```

**多个候选可以批量比较。** 每个候选构成独立的文本对；N 个候选产生 N 个分类结果，按 `BATCH_SIZE` 分批前向计算，不是逐项发送 HTTP 请求，也不把整个候选数组拼成一段提示词让模型生成 ID。分批前按 token 长度分组以减少填充，计算后恢复原来的对应关系。

NLI 不是通用语义相似度模型。这里将“新问题符合历史问题描述”作为匹配代理：

| `MATCH_MODE` | 评分方式 | 适用情况 |
| --- | --- | --- |
| `directional`，默认 | `P(entailment，新问题 → 历史描述)` | 历史描述是较短的风险模式，如“编译不通过” |
| `bidirectional` | 两个方向的蕴含概率取最小值 | 要求语义更接近等价；历史描述应有足够细节，计算量约增加一倍 |

候选满足 `score >= THRESHOLD` 且矛盾概率 `<= MAX_CONTRADICTION` 才能命中；双向模式使用两个方向矛盾概率的最大值。返回合格候选中分数最高的 ID，同分时按 ID 字典序取第一个，使结果不依赖候选顺序。默认阈值 `0.85`、矛盾上限 `0.10` **只是初始配置，不是已校准的准确率保证**，softmax 分数也不是“该问题确实为误判”的概率。

选择直接 Transformers 推理，是为了减少部署组件、直接保留 NLI 分类头，并支持 CPU、GPU 和可选量化。当前规模是数百条历史记录，不需要向量数据库。未把 vLLM、Ollama 或 SGLang 作为默认依赖；替换推理服务时必须验证该具体分类架构、标签顺序和模板，不能用普通 Qwen 聊天模型代替。

## 2. 目录

```text
issue_filter/
  __init__.py                Flask API 与错误处理
  config.py                  环境变量配置
  history.py                 历史库校验、关键词召回
  nli.py                     模型加载、分批分类
  matcher.py                 模式评分、阈值、最佳 ID 选择
data/
  historical_issues.json     历史误判库
  evaluation.example.jsonl   评估数据格式示例
scripts/
  download_model.py          只下载选定模型子目录
  evaluate.py                业务样本评估、阈值扫描
tests/                      业务测试、分类模型集成测试
compose.yaml                CPU 部署
compose.gpu.yaml            NVIDIA GPU 部署覆盖配置
```

## 3. 历史误判库

编辑 `data/historical_issues.json`，初始内容为：

```json
[
  {"id": "compile_issue", "key_words": ["编译"], "description": "编译不通过"},
  {"id": "no_init_issue", "key_words": ["NOT_INIT"], "description": "NOT_INIT 未定义风险"}
]
```

规则：

- 默认对每条历史记录，`key_words` 中**任意一个**字符串出现在新描述中就召回（OR）。如果业务要求同一条记录的全部关键词都出现，设置 `KEYWORD_MODE=all`（AND）。
- 使用字面子串匹配，不是正则表达式、分词或向量检索。默认区分大小写，`NOT_INIT` 与 `not_init` 不同；可设置 `KEYWORD_CASE_SENSITIVE=false`。
- 每条记录的关键词必须为非空字符串数组，`id` 必须唯一，描述不能空白。字段严格为 `id/key_words/description`。空历史数组允许使用。
- 启动时一次性加载并校验；修改 JSON 后重启服务生效。Docker 挂载整个 `data` 目录，文件被替换后重启也能读到新内容。
- 默认最多 1000 条，足够数百条的使用场景；更多记录可调 `MAX_CANDIDATES`，该配置同时限制历史库和单次候选数，避免静默丢弃候选。
- 第二个接口接受当前历史库的候选子集，校验对象内容与关键词规则。候选过期、未知、重复或被改写会返回 400，请重新调用第一个接口。

对于真实业务，建议将“编译不通过”逐步细化为项目内具体的误判模式。例如“将项目构建时生成的 NOT_INIT 宏误认为未定义”。只有笼统的“编译不通过”会覆盖很多真实编译缺陷，模型无法从这五个字恢复公司规范。每个项目/规范范围使用对应历史库；关键词应覆盖常用表达，否则第一阶段未召回的条目不会进入语义比较。

## 4. 资源与部署选择

模型原始权重约 **9.08 GB**，下载还包含 tokenizer 等文件。以下为容量规划估算，未在当前机器测得：

| 路径 | 资源规划参考 | 说明 |
| --- | --- | --- |
| GPU，不量化 | 建议先按 16 GB 以上显存规划，短文本、小 batch | `DTYPE=auto` 优先 BF16，不支持时选 FP16 |
| GPU，4bit NF4 | 可从 8–12 GB 显存、小 batch 试起 | 不能保证任意 8 GB 卡可运行；嵌入、视觉塔、临时激活等仍占资源 |
| CPU，FP32 | 建议至少 32 GB RAM，并保留系统余量 | 速度较慢，主要用于联调或低吞吐场景；默认不做 CPU 量化 |

Docker 镜像、模型原始文件、缓存和本地副本需要额外磁盘空间，建议预留至少 25–35 GB。4bit 是加载时量化，**首次仍需下载原始权重**，也不保证一定比 BF16 更快。更换精度、模型大小或版本后应重新校准阈值。

### 4.1 推荐：Docker + NVIDIA GPU

需要 Docker Engine/Desktop、Compose v2、兼容 CUDA 12.8 的 NVIDIA 驱动和 GPU 容器支持。Linux 通常需要 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)；Windows 使用 Docker Desktop 的 Linux 容器/WSL2 GPU 支持。先确认 `docker info` 能连接引擎，并按 [Docker GPU 文档](https://docs.docker.com/compose/how-tos/gpu-support/) 确认容器可访问显卡。

在项目根目录执行：

```bash
cp .env.example .env
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml up -d
docker compose -f compose.yaml -f compose.gpu.yaml logs -f issue-filter
```

PowerShell 将复制命令替换为 `Copy-Item .env.example .env`。首次启动需要联网下载指定模型到持久化缓存，启动预热完成后才监听端口；后续启动复用缓存。需要代理或 Hugging Face 镜像时按部署环境配置；这不改变代码默认使用的官方模型来源。

显存不足时在 `.env` 修改：

```dotenv
QUANTIZATION=4bit
BATCH_SIZE=1
```

然后重新执行上述 `up -d`，使环境配置生效。GPU 镜像已包含 bitsandbytes；支持 `none/4bit/8bit`。分类 `score` 层保留浮点。没有适合的 CUDA 设备会明确失败，不会悄悄切换到 CPU。

健康检查：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

`/healthz` 检查进程，包含 `model_loaded` 与历史条目数；`/readyz` 在成功完成一次预热/推理后返回 200，后续模型异常会将其置为 503，直到推理恢复。参数错误不影响就绪状态。默认 `.env` 启用 `PRELOAD_MODEL=true`，启动时执行一次短文本推理。Docker healthcheck 使用 liveness；生产流量入口可检查 `/readyz`；实例异常后可先重启预热再接流量。下载较慢时可先按下节预下载，避免启动等待过久。

### 4.2 Docker CPU

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f issue-filter
```

基础 Compose 强制 `DEVICE=cpu`、`QUANTIZATION=none`，安装 CPU 版 PyTorch；GPU 覆盖文件才启用 CUDA，避免 CPU 环境下载不必要的 CUDA 包。模型文件和缓存是两个命名卷，重建容器不会删除它们。

### 4.3 预下载与离线运行

GPU 方案构建后，联网下载到模型卷（CPU 部署去掉 `-f` 两个参数即可）：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm issue-filter python -m scripts.download_model --output /models
```

脚本只下载指定子目录，不会下载 openjev 仓库的其他模型。完成后修改 `.env`：

```dotenv
MODEL_PATH=/models/qwen3.5-4b-nli-v5
LOCAL_FILES_ONLY=true
HF_HUB_OFFLINE=1
```

再次 `up -d`。此时只读取本地 checkpoint，不再访问模型仓库。`MODEL_PATH` 应直接指向含 `config.json`、tokenizer 文件和 safetensors 的目录；配置它后忽略 `MODEL_ID/MODEL_SUBFOLDER/MODEL_REVISION`。

离线服务器还需要事先导入构建好的镜像和模型卷数据，或将 checkpoint 目录只读挂载到上述 `/models/...` 路径。只复制 `.env` 和 Python 代码不能完成离线部署。在线首次部署也不会把待审核的问题描述发送到远程模型 API；业务推理始终在本地进程执行。

### 4.4 不使用 Docker

```bash
python -m venv .venv
# Linux/macOS shell
source .venv/bin/activate
# Windows PowerShell 使用：.\.venv\Scripts\Activate.ps1

# 按设备二选一，先安装正确的 PyTorch wheel
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU 改为：python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-model.txt
# 仅 GPU 量化需要：python -m pip install -r requirements-quantization.txt

python -m scripts.download_model --output models
```

原生 Python 启动**不会自动读取 `.env`**，需要设置进程环境变量。PowerShell 示例：

```powershell
$env:MODEL_PATH=(Resolve-Path '.\models\qwen3.5-4b-nli-v5').Path
$env:LOCAL_FILES_ONLY='true'
$env:DEVICE='cpu'  # GPU 时设置为 cuda
$env:PRELOAD_MODEL='true'
waitress-serve --listen=127.0.0.1:8000 --threads=4 wsgi:app
```

Linux GPU 示例：

```bash
export MODEL_PATH="$PWD/models/qwen3.5-4b-nli-v5"
export LOCAL_FILES_ONLY=true DEVICE=cuda PRELOAD_MODEL=true
gunicorn --config gunicorn.conf.py wsgi:app
```

进程启动前检查 Python 依赖与硬件。只想联调第一个接口时，可只安装 `requirements.txt` 并设置 `PRELOAD_MODEL=false`；此时不需要模型，第二个接口的非空候选推理仍需要完整依赖和权重。

## 5. API 使用

所有请求使用 `Content-Type: application/json`。如果 `.env` 配置了 `API_KEY`，两个业务接口还需要 `Authorization: Bearer <API_KEY>`；健康检查不需要认证。Compose 默认只绑定本机 `127.0.0.1`，需要供其他机器调用时可通过公司内网网关暴露服务并配置鉴权。

### 5.1 召回候选

```bash
curl -X POST http://127.0.0.1:8000/api/v1/potential-issues \
  -H 'Content-Type: application/json' \
  -d '{"description":"NOT_INIT 未定义，可能导致编译不通过。"}'
```

返回：

```json
[
  {"id": "compile_issue", "key_words": ["编译"], "description": "编译不通过"},
  {"id": "no_init_issue", "key_words": ["NOT_INIT"], "description": "NOT_INIT 未定义风险"}
]
```

无关键词命中返回 `[]`。该接口只做字符串匹配，不加载模型、不执行语义判断。

### 5.2 语义匹配

```bash
curl -X POST http://127.0.0.1:8000/api/v1/match \
  -H 'Content-Type: application/json' \
  -d '{
    "description":"NOT_INIT 未定义，可能导致编译不通过。",
    "potential_historical_issue_list":[
      {"id":"compile_issue","key_words":["编译"],"description":"编译不通过"},
      {"id":"no_init_issue","key_words":["NOT_INIT"],"description":"NOT_INIT 未定义风险"}
    ]
  }'
```

命中时的**结构示例**（具体 ID 由实测模型分数决定，不承诺此输入固定返回某个 ID）：

```json
{"matched_issue_id": "no_init_issue"}
```

全部候选低于阈值、矛盾概率过高或候选数组为空时：

```json
{"matched_issue_id": null}
```

空数组直接返回，不加载模型。候选来自第一个接口即可；不用客户端逐个调用模型。

### 5.3 Python 完整调用（仅标准库）

```python
import json
import os
from urllib.request import Request, urlopen

base_url = "http://127.0.0.1:8000"

def post(path, payload):
    headers = {"Content-Type": "application/json"}
    if os.getenv("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
    req = Request(base_url + path, data=json.dumps(payload).encode("utf-8"),
                  headers=headers, method="POST")
    with urlopen(req, timeout=300) as response:
        return json.load(response)

description = "NOT_INIT 未定义，可能导致编译不通过。"
candidates = post("/api/v1/potential-issues", {"description": description})
result = post("/api/v1/match", {
    "description": description,
    "potential_historical_issue_list": candidates,
})
print(result["matched_issue_id"])  # str 或 None
```

### 5.4 错误约定

```json
{"error": {"code": "model_unavailable", "message": "模型加载或推理失败；请检查依赖、权重、设备和内存"}}
```

| 状态码 | 场景 | 客户端处理 |
| --- | --- | --- |
| 400 | JSON、字段或描述无效；候选重复、过期、未知；字符/候选数超限 | 修正输入或重新召回 |
| 401 | API key 缺失或无效 | 携带正确认证信息 |
| 413 | 请求体超过默认 4 MiB | 减少请求数据 |
| 415 | 非 JSON 请求 | 使用正确 Content-Type |
| 422 | 完整 NLI 文本对超过 `MAX_LENGTH` tokens | 缩短问题描述；不得当作无匹配 |
| 503 | 模型不可用、显存不足或推理锁等待超时 | 保留原审核问题；按 Retry-After 重试或排查服务 |

不静默截断文本，避免丢掉末尾否定词或历史描述后得出错误结论。`null` 只表示成功完成判断后没有合格匹配；基础设施错误不会伪装成 `null`。服务不在日志中记录请求正文。

## 6. 配置与性能

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HISTORY_PATH` | `data/historical_issues.json` 的项目绝对路径 | 历史库位置 |
| `MODEL_ID` / `MODEL_SUBFOLDER` | `AlexWortega/openjev` / `qwen3.5-4b-nli-v5` | 模型来源 |
| `MODEL_REVISION` | 上文的固定 commit SHA | 推荐固定版本 |
| `MODEL_PATH` | 空 | 本地 checkpoint，优先级最高 |
| `LOCAL_FILES_ONLY` | `false` | 禁止模型加载时下载 |
| `DEVICE` | 原生运行 `auto`；Compose 按部署方案覆盖 | `auto/cpu/cuda` |
| `DTYPE` | `auto` | `auto/float32/float16/bfloat16`，CPU 支持 FP32 |
| `QUANTIZATION` | `none` | CUDA 可选 `4bit/8bit` |
| `BATCH_SIZE` | `4` | 每次前向计算的文本对数，显存不足先减为 1 |
| `MAX_LENGTH` | `2048` | 模板 + 两段描述的完整 token 上限 |
| `MATCH_MODE` | `directional` | 也支持 `bidirectional` |
| `THRESHOLD` | `0.85` | 蕴含匹配阈值 |
| `MAX_CONTRADICTION` | `0.10` | 矛盾概率上限 |
| `KEYWORD_MODE` | `any` | `any/all` |
| `KEYWORD_CASE_SENSITIVE` | `true` | 关键词是否区分大小写 |
| `MAX_DESCRIPTION_CHARS` | `8000` | 每条新/历史描述字符上限，仍受 token 上限约束 |
| `MAX_CANDIDATES` | `1000` | 历史库及请求候选数上限 |
| `MAX_CONTENT_LENGTH` | `4194304` | 请求体字节上限 |
| `INFERENCE_LOCK_TIMEOUT` | `1` | 等待本进程模型推理锁的秒数 |
| `PRELOAD_MODEL` | Python 默认 `false`；示例 `.env` 为 `true` | 启动时加载并预热 |
| `API_KEY` | 空 | 可选 Bearer 认证 |
| `PORT` | `8000` | Compose 的宿主机端口 |
| `GUNICORN_TIMEOUT` | `600` | Gunicorn worker 存活监测超时，不是每个请求的硬截止时间 |

服务使用一个 Gunicorn 进程、四个线程，一份模型常驻内存；不要直接增加 GPU worker 数，否则会复制权重。模型推理通过锁串行调度，请求内批处理；它不是跨请求动态批处理。关键词查询不获取模型锁。处理大量候选时应结合实际长度测量时延和显存，再逐步增大 batch。

默认使用 PyTorch SDPA，无需额外编译 FlashAttention。Qwen3.5 的线性注意力路径可能使用较慢的 PyTorch 回退实现，不能仅凭参数量估算吞吐。高吞吐时再针对目标硬件测试兼容的 `causal_conv1d/fla` 内核或专门的分类推理服务，当前镜像不自动编译这些可选内核。

更新权重：修改模型子目录和 revision，预下载后重新部署，并重新跑评估。当前代码会核验三分类标签和权重加载结果，发现缺失分类头或形状不匹配时拒绝运行；它不会用随机分类头继续提供结果。小模型中文和代码风险描述的效果仍需业务样本验证，作者展示的通用 NLI 成绩不能直接推导为公司代码审核误判识别准确率。

## 7. 测试与业务评估

### 7.1 无模型业务测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q -m 'not model_stack and not real_model'
python -m ruff check .
```

覆盖关键词 OR/AND、大小写和字面匹配、500 条历史记录、阈值与最佳候选、同分稳定性、双向评分、空列表不加载模型、参数校验、鉴权、并发繁忙以及模型错误不变成无匹配等。测试用替身产生指定分类概率，用于验证业务流程，不代表真实模型语义质量。

### 7.2 Transformers 调用链测试

安装 `requirements-model.txt` 后：

```bash
python -m pytest -q -m model_stack
```

测试在临时目录构造微型随机 Qwen3.5 分类权重和 tokenizer，实际运行混合注意力模型，验证本地加载、padding ID 同步、标签映射及分批结果与逐条结果一致；无需下载大模型。这仍然不代表真实 4B checkpoint 的业务效果。

### 7.3 真实模型冒烟

下载权重后设置 `MODEL_PATH` 并启用：

```bash
RUN_REAL_MODEL_TESTS=1 MODEL_PATH="$PWD/models/qwen3.5-4b-nli-v5" python -m pytest -q -m real_model
```

PowerShell：

```powershell
$env:RUN_REAL_MODEL_TESTS='1'
$env:MODEL_PATH=(Resolve-Path '.\models\qwen3.5-4b-nli-v5').Path
python -m pytest -q -m real_model
```

仅检查相同语句和否定语句的基本关系；默认跳过，明确要求本地权重且禁止自动下载。若该项失败，应先排查加载/精度和模型对中文的表现，再考虑接入过滤流程。

### 7.4 校准业务阈值

准备 JSONL，每行包含一个描述和人工标注的 ID；真正的问题或不属于库中误判模式的问题标注 `null`：

```json
{"description":"编译无法通过。", "expected_id":"compile_issue"}
{"description":"编译可以通过，但运行时存在资源泄漏。", "expected_id":null}
```

运行（需完整模型依赖与权重）：

```bash
python -m scripts.evaluate data/evaluation.example.jsonl --thresholds 0.75,0.80,0.85,0.90,0.95
```

Docker 示例，先将业务样本放入 `data` 目录：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm issue-filter python -m scripts.evaluate /app/data/evaluation.example.jsonl
```

输出 `keyword_recall`、各阈值的 `id_precision/id_recall/exact_accuracy`、`false_filters`（真实问题被过滤数）和 `wrong_ids`（命中错误历史 ID 数）。同一条样本只推理一次，阈值扫描复用分数；没有预测正例时 precision 返回 `null`。示例文件只有格式演示作用，不是评估集，也未声称模型能全部通过。

应使用独立的校准集和测试集，加入同关键词不同根因、否定、不同对象、跨项目规则差异和新增代码标识符等难负例，避免同一问题的改写同时进入两组数据。优先降低 `false_filters`，确认阈值后再上线。单条新描述如果包含多个独立风险，应由上游拆成逐条问题；否则某一部分匹配到历史模式可能掩盖剩余的新风险。

返回匹配 ID 表示“可能复用了某种历史误判”，不能证明当前代码没有缺陷。上游应保存原始问题、历史 ID、模型 revision、匹配模式和阈值，便于审计和抽样复核。

### 7.5 本次实现的验证范围

2026-11-04 在 Windows / Python 3.11 环境验证：`pytest` 为 **48 passed, 1 skipped**；通过 Ruff 静态检查、pip 依赖一致性检查、CPU/GPU Compose 配置解析，以及真实 Waitress 本机 HTTP 调用。微型随机 Qwen3.5 的 Transformers 调用链测试已通过。

跳过项是真实 4B checkpoint 语义冒烟；本机 Docker 引擎未运行，未发现 NVIDIA 工具，因此没有启动 GPU 容器，也没有下载或实测 9.08 GB 权重。真实模型中文准确率、量化兼容性、显存和吞吐仍需按上述步骤在目标硬件上验证。

## 8. 故障排查

- `docker ... dockerDesktopLinuxEngine ... not found`：先启动 Docker Desktop 并使用 Linux 容器。
- `model_unavailable`：检查依赖是否按文件安装、模型路径/缓存是否完整、GPU 是否可用、内存是否足够。API 隐去底层细节；可在相同环境执行下面的本地诊断得到加载栈，不会发送业务描述：

  ```bash
  python -c "from issue_filter.config import Settings; from issue_filter.nli import TransformersNLI; TransformersNLI(Settings.from_env()).load()"
  ```

- `model_busy`：当前请求正在推理；按 `Retry-After` 重试，减少调用方并发或扩容独立实例。
- CUDA OOM：先将 `BATCH_SIZE` 减为 1，使用 4bit 或更大显存设备。单条文本仍过长时缩短描述。部署改变后重新评估。
- `input_too_long`：整对文本超过 token 限制，不是字符数限制；保持每个审核问题聚焦单个风险。
- pip 镜像返回 403：检查组织镜像是否可用；允许访问官方源时可在安装命令追加 `--index-url https://pypi.org/simple`。
- 修改 `.env` 后用 `docker compose ... up -d` 重建容器配置；只改历史 JSON 时 `restart issue-filter` 即可。CPU/GPU 命令须始终使用对应的 Compose 文件组合。

## 9. 依据

- [模型卡与推荐 checkpoint](https://huggingface.co/AlexWortega/openjev)
- [v5 模型配置、标签和输入模板](https://huggingface.co/AlexWortega/openjev/blob/main/qwen3.5-4b-nli-v5/config.json)
- [原始权重文件及大小](https://huggingface.co/AlexWortega/openjev/blob/main/qwen3.5-4b-nli-v5/model.safetensors)
- [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [bitsandbytes 量化文档](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [PyTorch 2.10.0 官方安装命令](https://pytorch.org/get-started/previous-versions/)

实现使用固定的直接依赖版本；间接依赖由 pip 解析。生产发布应在目标平台构建和测试镜像后记录其 digest，并归档业务评估结果。
