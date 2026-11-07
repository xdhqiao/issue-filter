# 内网离线部署（Linux x86_64 / CPU）

本包可在内网加载运行，不需要访问 Docker Hub、PyPI 或 Hugging Face。Python 3.15.2、Flask、Gunicorn、PyTorch CPU、Transformers、Accelerate 及其依赖已安装在镜像中；约 9.1 GB 的 Qwen3.5-4B-NLI-v5 权重、tokenizer 和配置也已封入同一镜像。

## 包含内容

| 文件 | 用途 |
| --- | --- |
| `images/issue-filter-offline-20261105-py315-cpu.tar` | 可直接 `docker load` 的完整应用镜像，包含全部基础层和模型 |
| `host-tools/docker-29.9.3.tgz` | 可选的 Linux x86_64 Docker Engine 静态安装包，含 containerd/runc |
| `host-tools/docker-compose-linux-x86_64` | Compose v5.7.0 可执行文件，可直接使用或安装为 CLI 插件 |
| `load-and-start.sh` / `.ps1` | 校验、导入镜像并离线启动 |
| `check-api.sh` / `.ps1`、`check_api.py` | 等待模型就绪并验证两个 API |
| `install-docker-linux.sh` | Linux 主机没有 Docker 时的可选安装助手 |
| `compose.yaml`、`.env.offline` | 离线启动配置，无 build 步骤，禁止自动拉取镜像 |
| `data/historical_issues.json` | 可编辑的历史误判库 |
| `python-packages.txt` | 镜像中实际安装的全部 Python 包版本 |
| `manifest.json`、`SHA256SUMS` | 镜像 ID、模型版本、文件大小和校验值 |
| `verification.json` | 本次实际离线验证的记录和范围 |

最终运行只需要这个应用镜像。不必再分别导入 Python 基础镜像或运行额外的模型服务，基础层已经随镜像保存。

## 目标机器要求

- Linux x86_64，运行 Linux amd64 容器。不能直接用于 ARM64、国产其他指令集或 Windows 原生容器。
- CPU 路径不要求 NVIDIA 显卡，不需要 CUDA、bitsandbytes、FlashAttention 或 FLA 插件。
- 建议 32 GB 及以上内存，并为程序留出约 20–24 GB 可用空间；模型在 CPU 上按 FP32 加载。32 GB 指物理内存，不代表被限制为 16 GB 的 Docker 虚拟机够用。
- 建议为交付 tar、展开后的镜像 tar 和 Docker 镜像层预留至少 40 GB 磁盘；长期保留原模型或构建缓存需要更多空间。
- 主机已有可用 Docker 最方便。无 Docker 时附带的静态安装工具还要求主机内核支持 namespaces/cgroups/overlayfs，并已具备 systemd、iptables 和常用基础命令。这些是操作系统组件，不能靠应用镜像安装进宿主内核；缺失时先使用对应 Linux 发行版的离线安装介质补齐。
- Windows 可以使用此镜像，但须已安装并启用 WSL2/Linux 容器模式的 Docker Desktop，给 Docker 虚拟机分配足够内存；包中的 Docker Engine 安装工具是 Linux 版，不是 Windows Desktop 安装器。

## 1. 将完整 tar 拷贝到内网并解压

把 `issue-filter-offline-linux-amd64-cpu-py315-20261105.tar` 和同名 `.tar.sha256` 一起拷贝。单个文件大于 4 GB，移动介质需使用支持大文件的文件系统。

Linux：

```bash
sha256sum --check issue-filter-offline-linux-amd64-cpu-py315-20261105.tar.sha256
tar -xf issue-filter-offline-linux-amd64-cpu-py315-20261105.tar
cd issue-filter-offline-linux-amd64-cpu-py315-20261105
```

Windows PowerShell：

```powershell
Get-FileHash .\issue-filter-offline-linux-amd64-cpu-py315-20261105.tar -Algorithm SHA256
Get-Content .\issue-filter-offline-linux-amd64-cpu-py315-20261105.tar.sha256
tar -xf .\issue-filter-offline-linux-amd64-cpu-py315-20261105.tar
Set-Location .\issue-filter-offline-linux-amd64-cpu-py315-20261105
```

比较两个 SHA256 是否相同。希望检查解压后的所有初始文件时，在修改配置前运行 `sha256sum --check SHA256SUMS`；启动脚本只校验镜像和主机安装工具，允许后续修改 `.env.offline` 与历史库。

## 2. 安装宿主 Docker（已安装可跳过）

Linux systemd 主机：

```bash
sudo bash install-docker-linux.sh
```

脚本只使用包内文件，不执行 apt/yum/pip 或联网下载。它不会覆盖已经存在的独立 containerd/runc 二进制。原来已安装 Docker 时保留现有引擎，只在缺少 Compose 时安装插件。

已有 Docker 但没有 Compose 时，也可以不安装插件，直接使用包内可执行文件：

```bash
chmod +x host-tools/docker-compose-linux-x86_64
./host-tools/docker-compose-linux-x86_64 version
```

Windows 用户使用现有 Docker Desktop；本包没有安装 Windows 可选功能、WSL 内核或 Desktop 的权限和离线系统依赖。

## 3. 导入并启动

Linux（当前用户无 Docker 权限时使用 sudo）：

```bash
bash load-and-start.sh
bash check-api.sh
```

PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\load-and-start.ps1
powershell -ExecutionPolicy Bypass -File .\check-api.ps1
```

启动需要从本地权重加载模型并预热，可能等待数分钟。自检会等待 `/readyz`，然后检查编译类正例、NOT_INIT 正例以及否定负例，共调用两个业务 API 6 次。

只导入不启动：

```bash
bash load-and-start.sh --load-only
```

等效手动操作：

```bash
docker load --input images/issue-filter-offline-20261105-py315-cpu.tar
docker compose --env-file .env.offline -f compose.yaml -p issue-filter-offline up --pull never -d
```

脚本不执行 `docker build`，不要求内网访问镜像仓库。完整模型路径固定为 `/models/qwen3.5-4b-nli-v5`，并启用 `LOCAL_FILES_ONLY=true`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。

## 4. 访问与配置

默认接口地址：`http://127.0.0.1:8000`。

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl -X POST http://127.0.0.1:8000/api/v1/potential-issues \
  -H 'Content-Type: application/json' \
  -d '{"description":"NOT_INIT 未定义，可能导致编译不通过。"}'
```

两个业务 API 的参数见随包 `README.md`。若其他内网电脑需要访问，在 `.env.offline` 设置 `BIND_ADDRESS=0.0.0.0`，按需要设置 `API_KEY`；再执行：

```bash
docker compose --env-file .env.offline -f compose.yaml -p issue-filter-offline up --pull never -d
```

接口设置了 API key 后需要 `Authorization: Bearer <API_KEY>`。宿主防火墙需允许计划使用的端口。

历史库是外部挂载的 `data/historical_issues.json`。修改后重启：

```bash
docker compose --env-file .env.offline -f compose.yaml -p issue-filter-offline restart issue-filter
```

查看日志与停止：

```bash
docker compose --env-file .env.offline -f compose.yaml -p issue-filter-offline logs -f issue-filter
docker compose --env-file .env.offline -f compose.yaml -p issue-filter-offline down
```

## 5. 验证范围与模型来源

具体结果见 `verification.json`。它记录镜像 ID、容器网络模式、依赖检查、模型文件检查以及 HTTP 自检结果；以实测记录为准。测试没有在目标内网操作系统上安装宿主 Docker。GPU 版没有包含在此 CPU 包中，CPU 版不能通过仅添加 `--gpus all` 变成 CUDA 版。

模型为 `AlexWortega/openjev` 下的 `qwen3.5-4b-nli-v5`，固定 revision `5df9876e03397de71504a8f156509e6529c263d8`。权重官方 SHA256 与本地值在 `manifest.json` 记录；模型卡来源为 <https://huggingface.co/AlexWortega/openjev>。该社区 NLI 模型以三分类结果判断语义关系，不产生聊天文本。

`BENCHMARK.md` 是此前 120 条合成样本的 CPU/FP32 实测报告；它不是对任意内网项目准确率的承诺。更换历史库、模型、精度或阈值后需重新评估。

## 6. 仓库中的原始模型与交付文件

原始模型已经下载到仓库 `models/qwen3.5-4b-nli-v5/`。导入本包时不用另外拷贝该目录，因为同一份权重已在镜像内。镜像归档、完整交付 tar 位于仓库 `offline/`。这些大文件存在于本机仓库目录，但被 `.gitignore` 排除，避免直接提交数十 GB 的二进制；构建脚本、部署脚本和说明仍可随代码版本管理。

本包已升级到 Python 3.15.2 / PyTorch 2.15.0 CPU。旧的 Python 3.11 交付包仍单独保留；历史 120 条评估见 BENCHMARK.md，新运行时评估见 BENCHMARK-PY315.md。
