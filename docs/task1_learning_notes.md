# 任务一学习版说明：从全局理解 Qwen 推理服务部署

这份文档是给自己看的，不是提交版报告。它的目的不是把结果压缩得漂亮，而是帮你真的理解任务一在整个 Agent 实验里的位置、每个脚本为什么这么写、遇到问题时怎么判断，以及汇报时别人可能追问什么。

## 1. 先从全局看任务一

整份作业要跑通的是这条链路：

```text
本地模型服务 -> Python 客户端 -> ALFWorld 环境 -> ReAct 智能体循环 -> 轨迹日志 -> 错误分析和消融实验
```

任务一只负责第一段：本地模型服务。

这一步要证明三件事：

1. 本地模型能被加载起来。
2. 它不是只能在一个 Python 进程里手动调用，而是能以 HTTP 服务形式被别的脚本调用。
3. 服务有基本吞吐能力，后面跑 100 多条 ALFWorld 轨迹时不会每一步都慢到不可用。

所以任务一真正的关键词是“服务化”和“可复现”，不是单纯“模型能生成一句话”。

## 2. 为什么要单独建环境

题目在“常见坑”里说了：推理服务和 ALFWorld 客户端最好分两个环境，通过 HTTP 通信。这点非常重要。

原因是 vLLM 会绑定比较重的依赖，比如：

```text
vLLM -> PyTorch -> CUDA runtime wheels -> xformers/triton/ray/fastapi
```

ALFWorld 又依赖 TextWorld 等环境包。两个生态的依赖目标不一样，放在同一个 Conda 环境里很容易互相影响。比如 vLLM 可能需要较新的 torch，而 ALFWorld 的依赖可能对 Python 或某些包版本更敏感。

因此我们创建了：

```text
/root/autodl-tmp/envs/qwen-vllm-clean
```

这个环境只用于任务一的 Qwen/vLLM 服务。后面任务二会再建一个 ALFWorld 环境。

这里有个汇报时可以说的点：

> 我没有把模型推理和环境交互装在同一个 Conda 环境里，而是用 HTTP API 解耦。这样更接近真实 Agent 系统架构，也降低依赖冲突风险。

## 3. 一开始 Conda 为什么失败

最开始直接 `conda create` 失败，不是作业步骤错了，而是本机网络配置的问题。

本机 `.condarc` 里默认 Conda channel 是清华镜像：

```text
https://mirrors.tuna.tsinghua.edu.cn/anaconda/...
```

同时 shell 里有代理：

```text
HTTP_PROXY=http://127.0.0.1:7897
HTTPS_PROXY=http://127.0.0.1:7897
```

在这个组合下，访问清华镜像时出现：

```text
SSL unexpected eof while reading
```

但是实测官方 Anaconda 和官方 PyPI 都能正常访问。所以最终安装脚本使用：

```bash
conda create --override-channels -c defaults ...
python -m pip install --index-url https://pypi.org/simple ...
```

意思是：创建环境时临时绕开用户级 `.condarc`，显式使用官方源。

这个点可以作为“工程可复现性”的说明：报告里不需要大篇幅写踩坑，但你自己要知道，如果别人问“为什么不用默认 conda create”，答案是当前代理环境下默认清华源 SSL 不稳定，所以脚本显式覆盖 channel。

## 4. 任务一到底写了哪些代码

### 4.1 `scripts/setup_task1_env.sh`

这个脚本是我们自己写的，不是题目直接提供的。

题目提供的是参考命令，比如：

```bash
pip install vllm
vllm serve ./models/Qwen2.5-7B-Instruct ...
```

但正式交付需要一个可复现仓库，所以我们把这些手工步骤封装成脚本。

这个脚本做三件事：

1. 创建任务一独立环境。
2. 升级 pip。
3. 安装 `requirements-task1.txt`。

关键写法：

```bash
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/envs/qwen-vllm-clean}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_CHANNEL="${CONDA_CHANNEL:-defaults}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
```

这种写法的意思是：脚本有默认值，但你可以在命令行覆盖。例如：

```bash
ENV_PREFIX=/some/path PIP_INDEX_URL=https://pypi.org/simple bash scripts/setup_task1_env.sh
```

为什么这么写？因为复现实验经常要换机器、换路径、换镜像。把参数写死会让脚本脆弱；完全不写默认值又不方便。默认值 + 环境变量覆盖是比较稳妥的折中。

这里还有一处重要写法：

```bash
conda create -y \
  --override-channels \
  -c "${CONDA_CHANNEL}" \
  -p "${ENV_PREFIX}" \
  "python=${PYTHON_VERSION}"
```

`--override-channels` 是为了不使用 `.condarc` 里的清华镜像。`-p` 是用路径创建环境，而不是用名字创建环境，这样环境位置明确，仓库说明里也能直接写清楚。

### 4.2 `scripts/serve.sh`

这个脚本也是我们自己写的，但核心命令来自题目给的 vLLM 参考命令。

也就是说：题目给了“该怎么启动 vLLM”的命令模板，我们把它工程化成了一个可复现启动脚本。

核心参数：

```bash
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
```

为什么不直接写死一条命令？因为后续很可能要改端口、上下文长度或显存比例。例如调试 OOM 时可以这样：

```bash
MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.80 bash scripts/serve.sh
```

脚本里还用了：

```bash
exec "${VLLM_BIN}" serve "${MODEL_PATH}" ...
```

`exec` 的意思是用 vLLM 进程替换当前 shell 进程。好处是 Ctrl-C 会直接作用到 vLLM 服务，关闭更干净。

### 4.3 `scripts/check_llm.py`

这是最小验证脚本。它用 OpenAI SDK 请求本地 vLLM 服务：

```python
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
```

为什么 `api_key="EMPTY"`？因为本地 vLLM 默认不校验 API key，但 OpenAI SDK 参数上需要传一个值，所以用占位符。

为什么默认测 `/v1/completions`？因为后面 ReAct 原论文 prompt 是 completion 风格，而不是 chat 对话风格。题目常见坑也提醒：不要直接把 ReAct few-shot 硬塞进 chat template，否则 special tokens 和模板可能改变模型行为。

为什么脚本里用了 `httpx.Client(trust_env=False)`？因为服务器环境有 `HTTP_PROXY/HTTPS_PROXY`。OpenAI SDK 底层用 httpx，如果信任环境变量，连 `127.0.0.1` 都可能被送去代理，导致本地服务请求返回 502。本次排查中，`curl` 直连成功而 SDK 返回 502，原因就是这个代理行为。

所以这个写法是为了本地服务直连：

```python
http_client=httpx.Client(trust_env=False)
```

### 4.4 `scripts/benchmark_llm.py`

这个脚本用来做轻量吞吐测试。

它不是 vLLM 官方 benchmark，也不是最终 ALFWorld 成功率评测。它的作用是回答任务一要求的“实测吞吐”，并让我们提前理解串行请求和并发请求的差别。

核心思路：

```text
固定 prompt -> 发 N 个请求 -> 记录每个请求延迟和输出 token 数 -> 统计 tokens/s
```

并发使用：

```python
ThreadPoolExecutor(max_workers=concurrency)
```

为什么可以用线程？因为每个线程主要是在等 HTTP 响应，不是做大量 Python 计算。真正的模型推理在 vLLM 服务进程里。

## 5. 启动参数逐个理解

实际启动命令是：

```bash
vllm serve /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
```

`vllm serve`：启动 OpenAI-compatible HTTP 服务。

`MODEL_PATH`：本地模型路径。我们没有重新下载模型，因为 `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` 已经存在完整权重。

`--served-model-name qwen`：API 层对外使用的模型名。客户端以后只需要写 `model="qwen"`，不用关心真实模型目录。

`--host 0.0.0.0`：监听所有网卡。对于本机脚本，访问 `127.0.0.1` 即可；如果需要从外部访问，也有可能通过服务器端口访问。实际是否开放外部访问取决于服务器网络设置。

`--port 8000`：端口。后面 OpenAI SDK 的 base url 是：

```text
http://127.0.0.1:8000/v1
```

`--tensor-parallel-size 1`：张量并行卡数。当前是一张 A800，所以是 1。如果多卡并行才会改成 2、4 等。

`--max-model-len 16384`：服务允许的最大上下文。ALFWorld 里 50 步轨迹加 few-shot prompt 可能比较长，所以 16K 比 8K 更稳。A800 80GB 显存足够。

`--gpu-memory-utilization 0.90`：vLLM 能使用的显存比例。不是说一定马上用满 90%，而是显存规划时按这个上限给权重、KV cache、临时 buffer 等分配空间。

## 6. “需要搞清楚的问题”逐个回答

### 6.1 `--max-model-len` 和显存占用是什么关系？为什么调大了会 OOM？

大模型生成时，每生成一个 token，都要看前面的上下文。如果每一步都重新计算完整上下文，代价太高。因此推理框架会保存每一层 attention 的 key/value，也就是 KV cache。

上下文越长，需要保存的历史 token 越多，KV cache 越大。并发请求越多，同时保存的序列越多，KV cache 也越大。

大致公式：

```text
KV cache ≈ 层数 × KV heads × head_dim × 2(K,V) × token数 × 并发序列数 × 每元素字节数
```

Qwen2.5-7B-Instruct 的关键配置：

```text
num_hidden_layers = 28
hidden_size = 3584
num_attention_heads = 28
num_key_value_heads = 4
torch_dtype = bfloat16
```

head_dim 是：

```text
3584 / 28 = 128
```

bf16 每个元素 2 bytes，所以单 token KV cache 约：

```text
28 × 4 × 128 × 2 × 2 bytes = 57344 bytes ≈ 56 KiB/token
```

如果单条序列是 16K token：

```text
56 KiB × 16384 ≈ 896 MiB
```

这只是理论 KV 数据大小，实际 vLLM 还要考虑 block 管理、CUDA graph、activation peak、临时 buffer 等。所以 `max-model-len` 调大以后，KV cache 规划变大，显存不够就会 OOM。

### 6.2 KV cache 占了多少显存？`--gpu-memory-utilization` 调整后并发数怎么变？

本次 vLLM 启动日志给了非常有用的实测信息：

```text
total_gpu_memory 79.25GiB × gpu_memory_utilization 0.90 = 71.33GiB
model weights: 14.25GiB
non_torch_memory: 0.14GiB
PyTorch activation peak: 2.18GiB
reserved for KV Cache: 54.76GiB
Maximum concurrency for 16384 tokens per request: 62.58x
```

也就是说，在当前参数下，vLLM 认为可以把约 54.76GiB 显存给 KV cache。

`gpu-memory-utilization` 越大，vLLM 可用于 KV cache 的显存通常越多，可支持的并发或上下文越大。但不能无限调大，因为模型运行还需要临时显存、CUDA runtime、碎片余量。如果调到 0.98 这类很激进的值，可能启动能过，但请求时 OOM。

所以这个参数本质是在“吞吐/并发”和“稳定性”之间取平衡。

### 6.3 continuous batching 是什么？串行请求和并发请求差多少？

普通 batch 的想象是：凑齐一批请求，一起开始，一起结束。问题是 LLM 请求长度不同，有的先结束，有的还在生成，GPU 利用率会浪费。

vLLM 的 continuous batching 是动态合批：服务运行过程中，新请求可以加入，完成的请求可以退出。这样多个短请求能共享 GPU 计算，吞吐会更高。

ALFWorld 评测特别适合这个机制，因为它不是一次生成长文章，而是：

```text
观察 -> 请求模型输出一个动作 -> 环境执行 -> 新观察 -> 再请求模型
```

全量 unseen 集大约会产生上百局 × 最多 50 步，也就是几千次短请求。串行跑会让 GPU 大量时间只处理一个请求；并发跑能让 vLLM 把多个 episode 的请求合批。

我们实测：

| concurrency | wall time | completion tokens/s | avg latency | p95 latency |
|---:|---:|---:|---:|---:|
| 1 | 8.89s | 83.02 | 0.55s | 0.66s |
| 8 | 1.26s | 564.04 | 0.60s | 0.66s |
| 16 | 0.83s | 896.26 | 0.69s | 0.70s |

结论：并发 8/16 的总吞吐明显更高，墙钟时间明显更短。单请求延迟略微上升，这是因为请求排队和合批带来一点等待，但整体效率更好。

### 6.4 如果用 Qwen3，thinking 输出怎么办？

Qwen3 是混合思考模型，默认可能输出：

```text
<think>...</think>
Action: ...
```

在 ReAct/ALFWorld 里，这会污染动作解析。环境只接受类似：

```text
go to fridge 1
take apple 1 from countertop 1
```

如果输出里混入 `<think>`，解析器可能拿到错误动作，导致 `Nothing happens`。

解决办法：

1. 在 chat template 参数中设置 `enable_thinking=false`。
2. 在 prompt 中加入 `/no_think`。
3. 在动作解析器里防御性移除 `<think>...</think>`。

本任务选择 Qwen2.5-7B-Instruct，所以这次不用处理 Qwen3 thinking。但你汇报时可以说：如果换 Qwen3，应把 thinking 模式作为独立设置说明，因为它会影响动作格式、token 开销和评测公平性。

## 7. 吞吐测试指标怎么理解

我们保存的主要结果文件是：

```text
logs/benchmark_task1_generation.json
```

里面的字段含义如下。

`num_requests`：总请求数。本次是 16，也就是每个并发设置下发 16 个请求。

`concurrency`：客户端并发度。比如 8 表示同时最多有 8 个 HTTP 请求在飞。它不是 vLLM 的理论最大并发，而是我们压测脚本设置的客户端并发。

`max_tokens`：每个请求最多生成多少 token。本次稳定版 benchmark 用 64。

`wall_time_sec`：这一组请求从开始到全部结束的总时间。这个指标越低，说明这组 workload 跑得越快。

`completion_tokens`：这一组请求一共生成了多少输出 token。注意它只统计 completion tokens，不包括 prompt tokens。

`completion_tokens_per_sec`：输出吞吐，计算方式是：

```text
completion_tokens / wall_time_sec
```

这是报告里最常见的 tokens/s。但要注意，它是输出 token/s，不是总 token/s，也不是训练吞吐。

`latency_avg_sec`：单个请求平均延迟。它回答的是“平均一个请求多久返回”。

`latency_p50_sec`：中位数延迟。50% 请求的延迟不超过这个值。

`latency_p95_sec`：尾部延迟。约 95% 请求的延迟不超过这个值。这个指标用来观察有没有少数请求特别慢。因为本次只有 16 个请求，p95 只是粗略参考，不适合过度解读。

`samples`：保存了几个输出样例，用来确认模型确实在生成文本，不是压测脚本只算了空响应。

## 8. OpenAI-compatible 是什么

OpenAI-compatible 不是说请求真的发到 OpenAI，而是本地服务模仿 OpenAI API 的路径和 JSON 格式。

后续 Python 代码可以这样写：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
resp = client.completions.create(
    model="qwen",
    prompt=prompt,
    temperature=0,
    max_tokens=100,
)
```

好处是：评测脚本不关心底层是 vLLM、SGLang 还是远程服务，只要 base_url 和 model 名一致就能跑。

## 9. 为什么用 completion 而不是 chat

题目明确提醒：ReAct 原始 prompt 是 completion 形式，不要直接套 chat template。

completion 形式大概是：

```text
[几个示例]
Observation: ...
Thought: ...
Action: ...
Observation: ...
>
```

模型接着续写下一个动作。

chat 形式会变成：

```text
<|im_start|>system ...
<|im_start|>user ...
<|im_start|>assistant ...
```

这些 special tokens 和模板可能改变模型行为。如果后面把 ReAct 改成 chat prompt，那应该作为一个单独消融变量，而不是偷偷混进 baseline。

所以任务一里我们同时验证 completion 和 chat，只是为了确认服务兼容；正式任务三 baseline 优先用 completion。

## 10. 如何复现任务一

从零复现任务一：

```bash
cd /root/alfworld-agent
bash scripts/setup_task1_env.sh
bash scripts/serve.sh
```

另一个终端验证：

```bash
cd /root/alfworld-agent
/root/autodl-tmp/envs/qwen-vllm-clean/bin/python scripts/check_llm.py --api completions
/root/autodl-tmp/envs/qwen-vllm-clean/bin/python scripts/check_llm.py --api chat
```

跑吞吐：

```bash
/root/autodl-tmp/envs/qwen-vllm-clean/bin/python scripts/benchmark_llm.py \
  --num-requests 16 \
  --concurrency 1 8 16 \
  --max-tokens 64 \
  --output logs/benchmark_task1_generation.json
```

服务跑完以后用 Ctrl-C 停掉，释放 GPU。本次停止后，`nvidia-smi` 显示 0MiB 占用。

## 11. 汇报时可以怎么讲

你可以按这个逻辑讲任务一：

1. 我把 Qwen2.5-7B-Instruct 部署成了本地 OpenAI-compatible 服务，后续 Agent 通过 HTTP 调用模型。
2. 推理服务和 ALFWorld 环境分开建环境，避免 vLLM/PyTorch/CUDA 和 TextWorld 依赖冲突。
3. 本机 A800 80GB，模型权重占 14.25GiB，vLLM 在 0.90 显存利用率下给 KV cache 预留 54.76GiB。
4. `max-model-len=16384` 是为了容纳 ReAct few-shot 加多步轨迹，但上下文越长 KV cache 越大，过大会 OOM。
5. vLLM continuous batching 适合 ALFWorld 这种大量短请求场景。实测并发 16 的输出吞吐约 896 tokens/s，明显高于串行的 83 tokens/s。
6. 本机代理会影响 conda/pip 和本地 SDK 请求，所以脚本里显式使用官方源，并让本地 HTTP 请求绕过代理环境变量。

这几句话基本覆盖了任务一最可能被问到的点。
