# ALFWorld Agent Starter

本仓库用于完成“智能体入门题”：把本地 Qwen 推理服务、ALFWorld 文本环境、ReAct 智能体循环、消融实验和实验报告串起来。

当前已完成任务一至任务四。任务四覆盖题目候选方向 1 至 6 的单变量消融；第 7 个 Reflexion 会引入失败后的第二次尝试，改变单次评测协议，因此未纳入主实验表。

## 目录结构

```text
configs/        ALFWorld 配置、任务四实验矩阵、固定 game manifest、失败样本标注
prompts/        ALFWorld few-shot prompt 文件
scripts/        环境部署、服务启动、评测批处理、结果分析脚本
docs/           各任务提交版报告和学习版说明
logs/           任务一压测、任务二环境检查与手动试玩记录
results/        任务三基线结果和任务四正式消融结果
tests/          任务四关键逻辑单元测试
```

模型权重和 ALFWorld 数据集不放入 Git 仓库。默认路径如下：

```text
Qwen2.5-7B-Instruct: /root/autodl-tmp/models/Qwen2.5-7B-Instruct
ALFWorld data:       /root/.cache/alfworld
```

## 一次性复现总流程

先开两个终端。

终端 A：启动本地 Qwen/vLLM 服务。

```bash
cd /root/alfworld-agent
conda activate /root/autodl-tmp/envs/qwen-vllm-clean
bash scripts/serve.sh
```

终端 B：运行 ALFWorld 客户端、评测和分析。

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
python -m py_compile eval.py scripts/run_task4.py scripts/analyze_task4.py
```

如果只想基于仓库内已有正式结果重新生成任务四汇总表：

```bash
python scripts/analyze_task4.py --results-root results/task4_prompt_ablation
```

如果要从头复现任务四当前 25 次正式运行：

```bash
python scripts/run_task4.py --stage primary
python scripts/run_task4.py --stage replicate --experiments baseline act_only cot_then_act random_nonmatching_2shot mixed_6shot zero_shot summary_10
python scripts/analyze_task4.py --results-root results/task4_prompt_ablation
```

说明：`primary` 运行 11 个配置的 seed 0；第二条命令给 7 个已补充三种子的配置补跑 seed 1、2，因此总计 11 + 7×2 = 25 次完整运行。runner 会跳过已经完整存在的结果。

## Task 1: Qwen Serving

### 选择

- 模型：`/root/autodl-tmp/models/Qwen2.5-7B-Instruct`
- 后端：vLLM
- 服务协议：OpenAI-compatible HTTP API
- 服务名：`qwen`
- 默认端口：`8000`

选择 Qwen2.5-7B-Instruct 是因为本机已经有完整权重，A800 80GB 可以直接承载 bf16 7B 模型，而且它不像 Qwen3 默认混合思考模式那样容易在 ReAct 动作解析里混入 `<think>...</think>`。

### 安装

```bash
cd /root/alfworld-agent
bash scripts/setup_task1_env.sh
conda activate /root/autodl-tmp/envs/qwen-vllm-clean
```

### 启动服务

```bash
cd /root/alfworld-agent
bash scripts/serve.sh
```

等日志显示服务可用后，在另一个终端验证：

```bash
cd /root/alfworld-agent
python scripts/check_llm.py --api completions
python scripts/check_llm.py --api chat
```

### 压测

```bash
cd /root/alfworld-agent
python scripts/benchmark_llm.py --num-requests 16 --concurrency 1 8 16 --max-tokens 64 --output logs/benchmark_task1_generation.json
```

结果会写入 `logs/benchmark_task1_generation.json`。

### 任务一文件

- `scripts/setup_task1_env.sh`：创建任务一专用环境并安装 vLLM/OpenAI SDK。
- `scripts/serve.sh`：启动 OpenAI-compatible vLLM 服务。
- `scripts/check_llm.py`：最小可用性验证。
- `scripts/benchmark_llm.py`：简单串行/并发吞吐测试。
- `docs/task1_submission_report.md`：任务一提交/汇报版报告。
- `docs/task1_learning_notes.md`：任务一学习版说明。

## Task 2: ALFWorld Text Environment

### 安装

```bash
cd /root/alfworld-agent
bash scripts/setup_task2_env.sh
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
```

如果数据还没有下载，可以执行：

```bash
DOWNLOAD_DATA=1 bash scripts/setup_task2_env.sh
```

### 环境检查

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
python env_check.py --config configs/base_config.yaml --split eval_out_of_distribution
```

### 手动试玩

任务二要求手动玩通一局。开始记录前运行：

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
alfworld-play-tw
```

当前可提交的手动交互记录保存在 `logs/manual_play_task2_clean.log`。

### 任务二文件

- `configs/base_config.yaml`：ALFWorld 文本环境配置。
- `env_check.py`：最小环境 reset/step 验证，并统计 train/seen/unseen 可玩任务数。
- `scripts/setup_task2_env.sh`：创建或复用 `alfworld` 环境并安装依赖。
- `requirements-task2.txt`：任务二直接依赖。
- `requirements-task2.lock`：当前任务二环境锁定依赖。
- `docs/task2_submission_report.md`：任务二提交/汇报版报告。
- `docs/task2_learning_notes.md`：任务二学习版说明。
- `logs/env_check_task2.log`：任务二程序化环境验证输出。

## Task 3: ReAct Evaluation Baseline

### 依赖

任务三复用 `alfworld` 环境，并额外需要 OpenAI SDK 调用本地 vLLM 服务：

```bash
cd /root/alfworld-agent
conda activate alfworld
python -m pip install -r requirements-task3.txt
export ALFWORLD_DATA=/root/.cache/alfworld
```

### Prompt 与本地命令适配

任务三使用 ReAct 官方 ALFWorld few-shot prompt：

```text
prompts/alfworld_3prompts.json
```

每类任务取两个 `react_*` 示例，不向 prompt 提供完整 `admissible_commands`。

当前本地 ALFWorld grammar 中普通放置命令是 `move OBJECT to RECEPTACLE`，而官方 ReAct prompt 常写 `put OBJECT in/on RECEPTACLE`。`eval.py` 会在动作解析阶段把后者规范化为前者，以保证模型输出能匹配本地 TextWorld 命令规则。

### 小样本检查

启动任务一 vLLM 服务后，先运行：

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
python -m py_compile eval.py
python eval.py \
  --split eval_out_of_distribution \
  --limit 3 \
  --workers 1 \
  --output-dir results/task3_smoke \
  --max-steps 50 \
  --temperature 0 \
  --max-tokens 100
```

### 全量评测

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
python eval.py \
  --split eval_out_of_distribution \
  --workers 8 \
  --output-dir results/task3_react_baseline \
  --max-steps 50 \
  --temperature 0 \
  --max-tokens 100
```

正式结果保存在：

- `results/task3_react_baseline/trajectories.jsonl`
- `results/task3_react_baseline/summary.json`
- `results/task3_react_baseline/metrics_by_task.csv`
- `results/task3_react_baseline/failure_analysis.csv`

本次全量 unseen 正式结果：整体成功率 92/134 = 68.7%，平均环境动作数 17.01，非法动作率 32.2%，parse failure rate 5.2%。这里的 `--max-steps 50` 表示最多 50 次 ALFWorld `env.step`；`think:` 不消耗环境步。任务三提交版报告见 `docs/task3_submission_report.md`，学习版说明见 `docs/task3_learning_notes.md`。

## Task 4: Prompt Ablations and Failure Analysis

任务四实现题目候选方向 1 至 6，共 11 个唯一配置。所有配置都在同一个 unseen 134 局 manifest 上评测，公共 baseline 固定为任务三条件：ReAct、任务匹配 2-shot、completion API、不注入合法动作、原始失败反馈、自由文本、完整历史、`temperature=0`、`max_tokens=100`、最多 50 个环境动作。

### 消融实验名称

| 配置 key | 中文备注 | 唯一改变因素 |
| --- | --- | --- |
| `baseline` | 公共基线：ReAct + 任务匹配 2-shot | 无 |
| `act_only` | 方向1 推理策略：Act-only | 只输出动作，不允许 thought |
| `cot_then_act` | 方向1 推理策略：CoT-then-Act | 开局先生成一次高层计划，之后按 Act-only 执行 |
| `random_nonmatching_2shot` | 方向2 Few-shot：随机非匹配 2-shot | 示例数量仍为 2，但来自其他任务类型 |
| `mixed_6shot` | 方向2 Few-shot：六类混合 6-shot | 六类任务各取一条示例 |
| `zero_shot` | 方向2 Few-shot：0-shot | 去掉轨迹示例 |
| `admissible_actions` | 方向3 动作空间：注入合法动作列表 | 每回合追加当前 admissible commands |
| `explicit_failure_feedback` | 方向4 失败反馈：显式无效动作恢复提示 | 只改写 `Nothing happens.` 反馈 |
| `guided_json` | 方向5 输出结构：Guided JSON | 输出限定为 JSON thought/action 两字段 |
| `recent_10` | 方向6 轨迹压缩：最近 10 turns | 只保留最近 10 个模型输出/反馈块 |
| `summary_10` | 方向6 轨迹压缩：摘要 + 最近 10 turns | 用动态摘要保存较旧轨迹信息 |

配置文件 `configs/task4_experiments.yaml` 中已保留这些中文备注，分析脚本导出的 CSV 也包含 `display_name_zh` 和 `direction_zh` 列。

### 环境

ALFWorld 客户端继续使用独立 `alfworld` 环境：

```bash
cd /root/alfworld-agent
conda activate alfworld
python -m pip install -r requirements-task4.txt
export ALFWORLD_DATA=/root/.cache/alfworld
```

推理服务继续使用任务一环境。vLLM 0.6.6.post1 的 guided JSON 在本机锁定 `xgrammar==0.1.11`：

```bash
cd /root/alfworld-agent
conda activate /root/autodl-tmp/envs/qwen-vllm-clean
python -m pip install -r requirements-task4-vllm.txt
bash scripts/serve.sh
```

### 验证与复现

在 `alfworld` 环境中依次执行：

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
python -m py_compile eval.py scripts/run_task4.py scripts/analyze_task4.py
python -m unittest tests.test_task4
python scripts/run_task4.py --stage dry-run
python scripts/run_task4.py --stage smoke
python scripts/run_task4.py --stage primary
python scripts/run_task4.py --stage replicate --experiments baseline act_only cot_then_act random_nonmatching_2shot mixed_6shot zero_shot summary_10
python scripts/analyze_task4.py --results-root results/task4_prompt_ablation
```

如果只按题目原计划补跑“最好/最差”两组额外种子，可把 replicate 命令简化为：

```bash
python scripts/run_task4.py --stage replicate
```

### 当前正式结果

当前仓库保留 25 次正式完整运行：11 个配置的 seed 0，以及 `baseline`、`act_only`、`cot_then_act`、`random_nonmatching_2shot`、`mixed_6shot`、`zero_shot`、`summary_10` 的 seed 1、2。所有完整运行均为 134 个唯一 game、同一 manifest、六类任务齐全、`errors=0`。

seed 0 最好项为 `summary_10`，91/134 = 67.9%，与本次 baseline 基本持平，但非法动作率和 token 更低；最差为 `zero_shot`，8/134 = 6.0%。已补三种子的配置中，`baseline` 为 67.41% ± 0.43 个百分点，`summary_10` 为 67.16% ± 1.97 个百分点，`zero_shot` 为 6.22% ± 0.43 个百分点，`mixed_6shot` 为 54.48% ± 0.75 个百分点。

主要文件：

- `configs/task4_experiments.yaml`：实验矩阵、中文备注、假设、排序与固定 few-shot 映射。
- `configs/task4_eval_manifest.json`：134 局固定清单和集合 SHA-256。
- `scripts/run_task4.py`：批量运行、指定配置补种子、最好/最差复现。
- `scripts/analyze_task4.py`：一致性校验、聚合、配对分析和画图。
- `tests/test_task4.py`：任务四单元测试。
- `results/task4_prompt_ablation/runs/`：正式完整轨迹与指标。
- `results/task4_prompt_ablation/analysis/`：消融总表、分任务表、多种子表、token 表、配对变化、失败样本和图表。
- `docs/task4_submission_report.md`：任务四提交/汇报版。
- `docs/task4_learning_notes.md`：实现、概念、踩坑、复现和答辩学习版。

## 结果检查

重新生成任务四分析表后，可检查一致性：

```bash
cd /root/alfworld-agent
conda activate alfworld
python scripts/analyze_task4.py --results-root results/task4_prompt_ablation
```

关键输出应显示：

```text
best=summary_10 worst=zero_shot
Wrote Task 4 analysis to /root/alfworld-agent/results/task4_prompt_ablation/analysis
```

核心分析表：

- `results/task4_prompt_ablation/analysis/ablation_results_seed0.csv`
- `results/task4_prompt_ablation/analysis/metrics_by_task_all_seed0.csv`
- `results/task4_prompt_ablation/analysis/token_costs_seed0.csv`
- `results/task4_prompt_ablation/analysis/paired_success_changes.csv`
- `results/task4_prompt_ablation/analysis/multiseed_all_configs.csv`
- `results/task4_prompt_ablation/analysis/multiseed_by_task_all_configs.csv`

## GitHub 上传建议

推荐准备两个私有仓库：

1. `alfworld-agent`：提交版，只包含可复现代码、部署脚本、评测脚本、prompt、正式结果和提交版报告。
2. `alfworld-agent-learning`：学习全量版，保留当前工作区的学习版文档、探索记录、测试和额外材料。

本机已生成提交版目录时，可这样上传提交版：

```bash
cd /root/alfworld-agent-submission
git init
git add -A
git commit -m "Prepare ALFWorld agent submission"
git branch -M main
git remote add origin git@github.com:<your-name>/alfworld-agent.git
git push -u origin main
```

学习全量版可在当前仓库上传：

```bash
cd /root/alfworld-agent
git init
git add -A
git commit -m "Complete ALFWorld agent workspace"
git branch -M main
git remote add origin git@github.com:<your-name>/alfworld-agent-learning.git
git push -u origin main
```

如果已经配置 GitHub CLI，也可以把最后三步替换为：

```bash
gh repo create <repo-name> --private --source=. --remote=origin --push
```

上传前建议执行：

```bash
git status --short
git ls-files | grep -E '(__pycache__|\.pyc$|task3_smoke|task4_smoke|reference_baseline_probe)' || true
```

提交版不需要包含模型权重、conda 环境目录、ALFWorld 数据集缓存、smoke test 输出、`__pycache__`、`.pyc` 或临时探测结果。
