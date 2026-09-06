# 任务三学习版说明：ReAct baseline 全流程与结果复盘

## 1. 任务三到底在做什么

任务一解决“本地模型服务怎么被 Python 脚本调用”，任务二解决“ALFWorld 文本环境怎么 reset、step 和手动玩”。任务三把两者连起来：让模型反复读取环境 observation，基于 ReAct prompt 输出 thought 或动作，再由 ALFWorld 返回新 observation，直到任务成功、环境步数达到上限，或模型进入明显死循环。

这条链路可以理解成：

```text
ALFWorld observation
  -> ReAct few-shot prompt + 当前任务 + 最近轨迹
  -> Qwen /v1/completions 输出一行
  -> 解析为 thought / action / parse_failure
  -> action 才 env.step，thought 只记录 OK.
  -> 新 observation 继续拼回上下文
```

这里非常重要的一点是：`eval.py` 并不是任务拆解器。它不会手写“先找物体、再拿物体、再放置”的策略。任务拆解能力来自 ReAct few-shot 示例，模型根据示例自己学会怎么分解；代码只做工程执行：选 prompt、解析命令、调用环境、保存结果。

## 2. 需要提交什么，每个文件有什么用

| 文件 | 作用 | 汇报时可以怎么说 |
| --- | --- | --- |
| `eval.py` | ReAct baseline 主评测脚本。 | “它实现了模型调用、动作解析、本地 grammar 适配、ALFWorld 交互、并发评测和结果落盘。” |
| `prompts/alfworld_3prompts.json` | ReAct 官方 ALFWorld few-shot prompt。 | “任务三按要求使用官方 prompt，每类任务取 2 个示例。” |
| `results/task3_react_baseline/trajectories.jsonl` | 每局完整轨迹，一行一个 episode。 | “后续错误分析不靠感觉，而是逐步看 raw output、规范化动作和环境反馈。” |
| `results/task3_react_baseline/summary.json` | 机器可读的整体指标和分任务指标。 | “用于核对成功率、平均环境步数、非法动作率、parse failure 等。” |
| `results/task3_react_baseline/metrics_by_task.csv` | 表格版结果。 | “可以直接复制到实验报告。” |
| `requirements-task3.txt` / `requirements-task3.lock` | 任务三依赖记录。 | “ALFWorld 环境额外安装 OpenAI SDK 来访问本地 vLLM。” |
| `docs/task3_submission_report.md` | 提交/汇报版报告。 | “偏正式，写实验设置、结果表、错误分析和后续方向。” |
| `docs/task3_learning_notes.md` | 学习版说明。 | “解释每一步为什么这么做，方便自己答辩和继续做任务四。” |

## 3. prompt 示例动作到底符不符合规则

从 ReAct 官方 prompt 文件本身看，示例动作是规范的。检查 `prompts/alfworld_3prompts.json` 中所有 `react_*` 示例，共有 289 个 ReAct turn，其中 91 个 thought、198 个 action，没有解析失败。示例里的动作包括：

```text
go to ...
open ...
take ... from ...
clean ... with ...
heat ... with ...
cool ... with ...
use ...
look
put ... in/on ...
```

问题不在“prompt 有没有 ReAct 格式”，而在“prompt 的放置动作表达和当前本地 ALFWorld 版本是否一致”。本地规则文件在：

```text
/root/.cache/alfworld/logic/alfred.twl2
```

这个文件里普通放置动作是：

```text
move {object} to {receptacle}
```

而不是：

```text
put {object} in/on {receptacle}
```

所以官方 prompt 是任务三要求的 prompt，但它和当前本地 TextWorld grammar 在放置命令上有版本/表达差异。这个差异如果不处理，模型经常走到最后一步才失败。

## 4. 这个 bug 是怎么确认的

之前的正式结果只有 5/134，看起来离文档预期很远。抽失败轨迹后发现大量失败都卡在类似动作：

```text
put saltshaker 2 in/on cabinet 1
put apple 1 in/on garbagecan 1
put bowl 1 in/on cabinet 1
```

这些动作来自官方 prompt 的示例风格，模型学得没错；但当前环境不接受它。用本地环境实际验证：

```text
put saltshaker 2 in/on cabinet 1 -> Nothing happens
move saltshaker 2 to cabinet 1   -> reward=1, won=True
```

这说明低成功率不是模型完全不会任务，而是动作语言和环境 grammar 没对齐。这个问题非常典型：Agent 评测里 prompt、解析器、环境 grammar 三者必须逐项对齐，否则结果会被工程 bug 污染。

## 5. eval.py 现在怎么解析动作

`normalize_action()` 负责把模型输出的一行文本变成三类之一：

```text
thought：内部推理，不调用 env.step
action：候选环境动作，会调用 env.step
parse_failure：既不是 thought 也不是动作，不调用 env.step
```

动作规范化包括：

- 去掉 `Action:` / `Act:` 前缀。
- 去掉编号、引号、句号。
- 全部转小写。
- `go back to` -> `go to`。
- `pick up` / `pick` -> `take`。
- `put x in/on y`、`put x in y`、`put x on y`、`place x on y` -> `move x to y`。
- `turn on desklamp 1` / `switch on desklamp 1` -> `use desklamp 1`。
- 对自然语言包裹的输出，尝试抽取其中的候选动作。
- 对没有 `think:` 但明显是推理的话，归为 thought，而不是 parse failure。

这个改动不是“把 admissible commands 塞进 prompt”。模型没有看到合法动作列表，代码也没有从合法动作列表里替模型挑动作；这里只是把 ReAct prompt 的放置动作表达转换成本地 grammar 的等价表达。

## 6. 这几版到底做了什么

| 版本 | 主要变化 | 结果/现象 | 结论 |
| --- | --- | --- | --- |
| 旧版/初版 | 接近骨架写法，容易把模型每次输出都算作一步。 | 曾出现 6/134、平均步数接近 50 的旧表，但 `think:` 和环境步口径不够干净。 | 不能作为正式结果。 |
| V1：环境步修正版 | `--max-steps 50` 改为 50 次 `env.step`；区分 `agent_turns`、`env_steps`、`thought_steps`、`action_steps`；parse failure 与 invalid action 分开。 | 调试结果约 6/134；大量 episode 在最终放置动作附近失败。 | 计步正确了，但还没解决本地 grammar 兼容。 |
| V2：thought/parse loop 版 | 把无前缀自然语言归为 thought；增加 `thought_loop` 熔断。 | 正式结果一度为 5/134，parse failure 降低，但成功率仍异常低。 | 失败归因更清楚，但核心 bug 还在。 |
| V3：当前正式版 | 发现并修复放置命令差异：`put x in/on y` -> `move x to y`。 | 正式结果提升到 92/134，整体成功率 68.7%。 | 证明之前个位数成功率主要是 prompt 动作表达与本地 ALFWorld grammar 不一致导致的。 |

## 7. 50 步到底是什么意思

文档写“最大步数：50”，正确口径是最多 50 个 ALFWorld 环境动作，也就是最多 50 次：

```python
obs, reward, done, info = env.step([action])
```

ReAct 的 `think:` 不是环境动作。它只是模型在上下文里写下来的推理，脚本会给虚拟反馈 `OK.`。所以当前脚本里：

- 输出 `think:`：`thought_steps += 1`，`agent_turns += 1`，但 `env_steps` 不变。
- 输出合法动作格式：`action_steps += 1`，`env_steps += 1`，调用 `env.step`。
- 输出无法解析内容：`parse_failures += 1`，`agent_turns += 1`，不调用环境。

所以 `avg_steps` 等价于 `avg_env_steps`，代表平均真实环境动作数，不混入 thought。

## 8. parse failure 和 invalid action 的区别

`parse_failure` 是模型输出无法变成候选动作，例如：

```text
ouch
done
重新思考: ...
```

这类输出不会送进 ALFWorld。

`invalid_action` 是模型输出已经被解析成动作，并且调用了 `env.step`，但它不在当前环境的合法动作集合中。例如：

```text
examine alarmclock 1 with desklamp 1
use desklamp 1 on bowl 1
clean handtowel 1 with sinkbasin 1
```

当前定义是：

```text
invalid_action_rate = invalid_actions / action_steps
```

parse failure 既不进分子，也不进分母。

## 9. 上下文截断策略

ReAct prompt 本身很长，50 步轨迹继续拼接后可能爆上下文。本项目用 `--max-context-chars 24000` 做字符预算。注意它是字符数，不是 token 数。

截断策略是：

```text
固定保留：任务说明 + 官方 few-shot 示例 + 当前任务初始 observation
可截断：历史轨迹，从旧到新丢弃，只保留最近交互
```

本次正式全量结果里没有 episode 触发截断，最大上下文长度是 15200 字符。

## 10. 并发和异常处理

正式评测使用 8 个 worker。ALFWorld env 有内部状态，不能多个 worker 共用一个 env。当前脚本结构是：

```text
主进程收集 134 个 gamefile
切成 8 份
每个 worker 独立 load config、load prompt、init_env(batch_size=1)
worker 顺序跑自己的 gamefile
主进程汇总并排序写盘
```

如果 worker 初始化失败，会写 `worker_init_error`；如果单局 API 或 env 报错，会写 `runtime_error`。当前正式结果里 `errors=0`。

## 11. 如何亲手复现

先启动模型服务：

```bash
cd /root/alfworld-agent
bash scripts/serve.sh
```

进入 ALFWorld 环境：

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
```

语法检查：

```bash
python -m py_compile eval.py
```

小样本检查：

```bash
python eval.py \
  --split eval_out_of_distribution \
  --limit 3 \
  --workers 1 \
  --output-dir results/task3_smoke \
  --max-steps 50 \
  --temperature 0 \
  --max-tokens 100
```

正式全量：

```bash
python eval.py \
  --split eval_out_of_distribution \
  --workers 8 \
  --output-dir results/task3_react_baseline \
  --max-steps 50 \
  --temperature 0 \
  --max-tokens 100
```

跑完看：

```text
results/task3_react_baseline/trajectories.jsonl
results/task3_react_baseline/summary.json
results/task3_react_baseline/metrics_by_task.csv
```

## 12. 当前正式结果怎么看

| Task type | Episodes | Success | Success rate | Avg env steps | Avg agent turns | Avg thought turns | Invalid action rate | Parse failure rate | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 134 | 92 | 68.7% | 17.02 | 25.03 | 6.70 | 32.2% | 5.2% | 0 |
| `pick_and_place_simple` | 24 | 23 | 95.8% | 15.17 | 17.62 | 2.12 | 8.8% | 1.9% | 0 |
| `pick_clean_then_place_in_recep` | 31 | 21 | 67.7% | 19.77 | 23.97 | 3.16 | 50.4% | 4.3% | 0 |
| `pick_heat_then_place_in_recep` | 23 | 20 | 87.0% | 13.52 | 19.57 | 5.70 | 16.7% | 1.8% | 0 |
| `pick_cool_then_place_in_recep` | 21 | 18 | 85.7% | 16.33 | 19.67 | 3.33 | 48.1% | 0.0% | 0 |
| `look_at_obj_in_light` | 18 | 2 | 11.1% | 19.22 | 46.56 | 23.78 | 39.9% | 7.6% | 0 |
| `pick_two_obj_and_place` | 17 | 8 | 47.1% | 17.82 | 28.65 | 7.06 | 12.5% | 13.1% | 0 |

终止原因：

| Termination reason | Episodes | 含义 |
| --- | ---: | --- |
| `success` | 92 | 环境给出成功奖励并结束。 |
| `env_done_without_success` | 8 | 环境结束或步数耗尽，但没有成功奖励。 |
| `parse_failure_loop` | 18 | 连续多次输出无法规范化为 thought 或 ALFWorld 动作。 |
| `thought_loop` | 16 | 连续多轮只推理不行动，触发模型调用熔断。 |

这个结果说明：

- `errors=0`，不是 API 或环境异常。
- 134 个唯一 gamefile 都跑完了，六类任务都有统计。
- `avg_steps=17.01` 表示平均真实环境动作数，不包含 thought。
- parse failure rate 是 5.2%，主要剩余问题是 `ouch`、`describe ...`、`turn off ...` 等格式偏离。
- invalid action rate 是 32.2%，主要来自 look-at 动作语义、clean/cool 状态顺序和部分重复动作。

## 13. 失败轨迹里的主要错误

### 1. Look-at 任务语义错误

`look_at_obj_in_light` 仍只有 2/18。模型常输出：

```text
examine alarmclock 1 with desklamp 1
use desklamp 1 on bowl 1
describe alarmclock 1
```

这些不是当前环境中稳定可执行的成功动作。该类任务需要更明确地学会“找到并拿起目标物体，再使用可用台灯”。

### 2. 多次失败后的格式循环

剩余 parse failure 中最常见的是：

```text
ouch
ouch!
done
重新思考: ...
```

这说明模型在连续 `Nothing happens.` 后会模仿反馈或退出到非动作文本。

### 3. Clean/cool 的状态顺序问题

clean/cool 类虽然成功率已经明显提高，但仍会出现对错误物体或错误状态执行工具动作，或状态改变后没有正确放置。

### 4. Pick-two 的记忆负担

`pick_two_obj_and_place` 已从 0 成功提升到 8/17，但仍会忘记第二个物体、重复探索或在第一个物体放置后格式崩坏。

## 14. 汇报时可以这样回答

如果老师问“baseline 是否符合任务三要求”，可以说：

> 符合。使用的是 ReAct 官方 ALFWorld few-shot prompt，每类 2 个示例，评测 `eval_out_of_distribution` 全量，`temperature=0`、`max_tokens=100`、最大 50 个环境动作，没有把 admissible commands 放进 prompt。代码只做动作解析、本地 grammar 适配和结果记录。

如果问“为什么要把 put 改成 move”，可以说：

> 这是本地环境兼容，不是 prompt 优化。本地 `alfred.twl2` 中普通放置动作是 `move object to receptacle`，而官方 ReAct prompt 使用 `put object in/on receptacle`。两者语义等价，但当前环境只接受前者。如果不适配，大量轨迹会在最后一步被错误判成失败。

如果问“为什么之前只有 5/134，现在 92/134”，可以说：

> 之前的低成功率主要由动作 grammar 不一致造成。抽失败轨迹发现大量任务已经找到并拿起物体，但最后 `put ... in/on ...` 被环境返回 `Nothing happens.`。修复为 `move ... to ...` 后，放置类任务恢复正常，说明这是工程 bug，而不是模型突然变强。

## 15. 后续应该怎么改

下一步进入任务四，不应该继续偷偷改 baseline。建议按单变量消融做：

1. 针对 look-at 的任务语义提示。
2. 连续 `Nothing happens.` 后的失败反馈改写。
3. ReAct vs Act-only vs CoT-then-Act。
4. 是否注入 admissible actions。
5. JSON/guided decoding 格式约束。
6. 针对 pick-two 的简短状态记忆。
7. 多次运行报告均值和标准差。
