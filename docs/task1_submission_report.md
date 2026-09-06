# 任务一：部署 Qwen 推理服务

## 1. 目标

本任务部署一个 OpenAI-compatible 的本地 Qwen 推理服务，使后续 ALFWorld 智能体可以通过 HTTP 请求调用大模型。推理服务与环境交互脚本分离，后续 ReAct、Act-only、prompt 消融和并发评测都复用同一个服务端。

## 2. 硬件、模型与后端

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA A800 80GB PCIe |
| 显存 | 81920 MiB |
| Driver | 590.48.01 |
| CUDA driver reported | 13.1 |
| 模型 | Qwen2.5-7B-Instruct |
| 模型路径 | `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` |
| 模型精度 | bfloat16 |
| 推理后端 | vLLM 0.6.6.post1 |
| 服务环境 | `/root/autodl-tmp/envs/qwen-vllm-clean` |
| API 模型名 | `qwen` |
| 服务地址 | `http://127.0.0.1:8000/v1` |

选择 Qwen2.5-7B-Instruct 的原因是本机已有该模型权重，A800 80GB 能直接运行 7B bf16 模型；同时该模型不像 Qwen3 混合思考模型那样默认输出 `<think>...</think>` 段，能减少后续 ReAct 动作解析被污染的风险。

推理服务使用独立 Conda 环境，与后续 ALFWorld/TextWorld 环境分开。这样可以避免 vLLM/PyTorch/CUDA 依赖与 ALFWorld 依赖冲突，二者只通过 HTTP API 通信。

## 3. 启动方式

服务启动脚本为 `scripts/serve.sh`。核心命令如下：

```bash
vllm serve /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
```

参数含义：

| 参数 | 含义 |
|---|---|
| `--served-model-name qwen` | 对外暴露的模型名，客户端统一用 `model="qwen"` 请求 |
| `--port 8000` | 本地服务端口 |
| `--tensor-parallel-size 1` | 单卡推理，不启用张量并行 |
| `--max-model-len 16384` | 最大上下文长度，给 ALFWorld 长轨迹留空间 |
| `--gpu-memory-utilization 0.90` | vLLM 最多使用 90% GPU 显存，为运行时留余量 |

vLLM 启动日志显示模型加载成功，权重占用 14.25 GiB；在 `gpu-memory-utilization=0.90` 下，vLLM 规划 54.76 GiB 给 KV cache，并估计 16K token/request 下最大并发约 62.58x。

## 4. 连通性验证

使用 `scripts/check_llm.py` 调用本地服务。`/v1/completions` 和 `/v1/chat/completions` 均通过验证。

completion 接口输出示例：

```text
当然，我已经配置好可以被本地脚本调用了。
```

chat 接口输出示例：

```text
我已经准备好可以通过本地脚本调用来使用了。
```

`/v1/models` 返回模型 `qwen`，`max_model_len=16384`。因此该服务可被本地 Python 脚本通过 OpenAI-compatible API 调用。

## 5. 吞吐测试

测试脚本：`scripts/benchmark_llm.py`。

```bash
python scripts/benchmark_llm.py \
  --num-requests 16 \
  --concurrency 1 8 16 \
  --max-tokens 64 \
  --output logs/benchmark_task1_generation.json
```

结果如下：

| setting | concurrency | num requests | wall time / s | completion tokens/s | avg latency / s | p95 latency / s |
|---|---:|---:|---:|---:|---:|---:|
| serial | 1 | 16 | 8.89 | 83.02 | 0.55 | 0.66 |
| concurrent | 8 | 16 | 1.26 | 564.04 | 0.60 | 0.66 |
| concurrent | 16 | 16 | 0.83 | 896.26 | 0.69 | 0.70 |

结果表明，并发请求显著提升吞吐并降低整体墙钟时间。并发 16 相比串行，completion tokens/s 从 83.02 提升到 896.26；平均单请求延迟略升，但总耗时明显下降。这与 vLLM continuous batching 的预期一致。

## 6. 关键问题回答

`--max-model-len` 越大，vLLM 需要为更长上下文规划更多 KV cache。KV cache 保存每层 attention 的 key/value，显存近似随层数、KV heads、head_dim、token 数、并发序列数和 dtype 字节数线性增长。Qwen2.5-7B-Instruct 的单 token KV cache 约为 56 KiB；16K token 单序列理论上接近 896 MiB，实际还要加 block 管理、CUDA graph 和临时 buffer 等开销。上下文或并发过大时就可能 OOM。

`--gpu-memory-utilization` 控制 vLLM 可使用的显存比例。本实验中 0.90 对应约 71.33 GiB 可用显存，其中模型权重占 14.25 GiB，KV cache 预留 54.76 GiB。调高该值通常能容纳更多 KV cache block，提高长上下文或并发能力；但过高会挤压运行时余量，增加 OOM 风险。

continuous batching 指 vLLM 在生成过程中动态加入新请求、移除完成请求，使多个短请求共享 GPU 计算。ALFWorld 评测会产生大量短请求，因此并发请求比串行请求更能利用 GPU。实测也显示并发 8/16 的墙钟时间明显低于串行。

如果使用 Qwen3，需要处理默认 thinking 输出。常见做法是在 chat template 参数中设置 `enable_thinking=false`，或在 prompt 中加入 `/no_think`，并在动作解析器里防御性移除 `<think>...</think>`。本实验选择 Qwen2.5-7B-Instruct，因此正式 baseline 不涉及该问题。

## 7. 任务一交付

本任务交付文件包括：

- `scripts/setup_task1_env.sh`
- `scripts/serve.sh`
- `scripts/check_llm.py`
- `scripts/benchmark_llm.py`
- `requirements-task1.txt`
- `requirements-task1.lock`
- `logs/benchmark_task1_generation.json`
- `docs/task1_submission_report.md`

其中 `logs/benchmark_task1_generation.json` 保留正式吞吐测试数据；连通性探活、显卡状态快照等一次性中间输出不作为最终交付文件保留。

结论：任务一已完成。本地 Qwen2.5-7B-Instruct 已通过 vLLM 部署为 OpenAI-compatible 推理服务，并完成连通性验证和并发吞吐测试。
