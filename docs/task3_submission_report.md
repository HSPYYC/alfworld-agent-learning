# 任务三：ReAct baseline 评测结果

## 1. 实验目标

本任务实现一个 ReAct 智能体循环，将任务一部署的本地 Qwen 推理服务与任务二部署的 ALFWorld TextWorld 环境连接起来，在 `eval_out_of_distribution` 全量 unseen split 上记录成功率、分任务成功率、平均步数和错误轨迹。实验目标不是追求最高成功率，而是得到一份可复现、可审计的 baseline，为任务四的错误分析与 prompt 消融实验提供依据。

需要强调的是，`eval.py` 没有实现一个单独的“任务拆解器”。任务分解来自 ReAct few-shot 示例，模型根据示例自行产生 thought 和动作；代码负责选择任务类型对应的 prompt、解析模型输出、执行 ALFWorld 命令、记录轨迹与指标。

## 2. 基线设定

| 项目 | 设置 |
| --- | --- |
| 模型服务 | vLLM OpenAI-compatible API |
| 模型 | Qwen2.5-7B-Instruct |
| 服务模型名 | `qwen` |
| 评测环境 | ALFWorld `AlfredTWEnv` 文本环境 |
| 评测 split | `eval_out_of_distribution` |
| Episode 数 | 134 |
| Prompt | ReAct 官方 `alfworld_3prompts.json` |
| 每类示例数 | 2 个 `react_*` 示例 |
| 采样参数 | `temperature=0`, `max_tokens=100` |
| 最大步数 | 50 个 ALFWorld 环境动作，即最多 50 次 `env.step` |
| 模型调用上限 | 默认 `3 * max_steps = 150` 个 agent turn |
| 并发 | 8 个 worker，每个 worker 独立创建 ALFWorld env |
| 上下文策略 | 保留 few-shot、当前任务和最近轨迹；`max_context_chars` 是字符预算 |
| Admissible commands | 不放入 prompt，只用于统计非法动作 |
| 本地命令适配 | 将 ReAct 示例常见的 `put OBJECT in/on RECEPTACLE` 规范化为本地 grammar 接受的 `move OBJECT to RECEPTACLE` |

正式运行命令：

```bash
python eval.py \
  --split eval_out_of_distribution \
  --workers 8 \
  --output-dir results/task3_react_baseline \
  --max-steps 50 \
  --temperature 0 \
  --max-tokens 100
```

## 3. 本地动作规则与 prompt 差异

本地 ALFWorld 的动作规则来自：

```text
/root/.cache/alfworld/logic/alfred.twl2
```

其中普通放置动作的模板是：

```text
move {object} to {receptacle}
```

而 ReAct 官方 prompt 中大量示例使用：

```text
put {object} in/on {receptacle}
```

该差异会导致模型在完成前面步骤后，最后放置动作被环境拒绝，返回 `Nothing happens.`。实际验证中，`put saltshaker 2 in/on cabinet 1` 在当前环境下不是合法动作，而语义等价的 `move saltshaker 2 to cabinet 1` 可以直接得到 `reward=1, won=True`。

因此当前 `eval.py` 保留官方 prompt，不把合法动作列表提供给模型，但在动作规范化层做本地 grammar 兼容：

```text
put x in/on y -> move x to y
put x in y    -> move x to y
put x on y    -> move x to y
place x on y  -> move x to y
```

该处理不改变任务分解，也不注入 `admissible_commands`；它只是把官方 prompt 的动作表达适配到当前安装版本的 ALFWorld 文本命令。

## 4. 指标定义

`--max-steps 50` 明确定义为最多执行 50 个 ALFWorld 环境动作。只有解析为真实动作并调用 `env.step([action])` 时，`env_steps` 才增加。`think:` 是 ReAct 的内部推理，不消耗环境步数。

- `agent_turns`：模型调用次数，包括 thought、action、parse failure。
- `env_steps`：真实环境动作数，即 `env.step` 调用次数。
- `thought_steps`：模型输出 `think:` 或被规范化为 thought 的次数，不调用环境。
- `action_steps`：成功解析为候选 ALFWorld 动作并发送给环境的次数；正常情况下等于 `env_steps`。
- `avg_steps`：平均环境动作数，等同于 `avg_env_steps`，不混入 thought。
- `parse_failures`：模型输出既不是 thought，也不能规范化为候选 ALFWorld 命令；这类输出不送入环境。
- `invalid_actions`：已经解析为动作、也送入了 `env.step`，但不在当前 `info["admissible_commands"]` 中的动作。
- `invalid_action_rate`：`invalid_actions / action_steps`。parse failure 不进入这个分子，也不进入这个分母。

每一步轨迹都保存 `raw_output`、`normalized`、`kind`、`observation`、`agent_turn`、`env_step`、`invalid_action`、上下文长度等字段。API 或环境异常不会静默丢失，而是作为 error episode 写入结果。

## 5. 动作解析与上下文处理

动作规范化覆盖了常见格式噪声：`Action:` / `Act:` 前缀、行首编号、引号、句号、大小写、`pick up` / `pick` 到 `take`、`go back to` 到 `go to`、`put x in/on y` 到 `move x to y`、`turn on` / `switch on` 到 `use`。对于包含自然语言包裹的输出，脚本会尽量抽取其中的候选动作，但不会使用 `admissible_commands` 替模型选择动作。

上下文采用字符预算 `--max-context-chars`，不是 token 数。截断策略是固定保留任务说明、两条官方 few-shot 示例和当前任务初始 observation；只有轨迹历史会从旧到新截断，保留最近交互。本次全量运行没有 episode 触发上下文截断，最大上下文长度为 15200 字符，低于 24000 字符预算。

## 6. 全量结果

| Task type | Episodes | Success | Success rate | Avg env steps | Avg agent turns | Avg thought turns | Invalid action rate | Parse failure rate | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 134 | 92 | 68.7% | 17.02 | 25.03 | 6.70 | 32.2% | 5.2% | 0 |
| `pick_and_place_simple` | 24 | 23 | 95.8% | 15.17 | 17.62 | 2.12 | 8.8% | 1.9% | 0 |
| `pick_clean_then_place_in_recep` | 31 | 21 | 67.7% | 19.77 | 23.97 | 3.16 | 50.4% | 4.3% | 0 |
| `pick_heat_then_place_in_recep` | 23 | 20 | 87.0% | 13.52 | 19.57 | 5.70 | 16.7% | 1.8% | 0 |
| `pick_cool_then_place_in_recep` | 21 | 18 | 85.7% | 16.33 | 19.67 | 3.33 | 48.1% | 0.0% | 0 |
| `look_at_obj_in_light` | 18 | 2 | 11.1% | 19.22 | 46.56 | 23.78 | 39.9% | 7.6% | 0 |
| `pick_two_obj_and_place` | 17 | 8 | 47.1% | 17.82 | 28.65 | 7.06 | 12.5% | 13.1% | 0 |

正式全量评测用时约 10.22 分钟。结果文件包含 134 行 episode 轨迹，对应 134 个唯一 gamefile；六类任务均有统计，且 `trajectories.jsonl`、`summary.json`、`metrics_by_task.csv` 中的总数、成功数、步数和错误数一致。

终止原因如下：

| Termination reason | Episodes | 含义 |
| --- | ---: | --- |
| `success` | 92 | 环境给出成功奖励并结束。 |
| `env_done_without_success` | 8 | 环境结束或步数耗尽，但没有成功奖励。 |
| `parse_failure_loop` | 18 | 连续多次输出无法规范化为 thought 或 ALFWorld 动作。 |
| `thought_loop` | 16 | 连续多轮只推理不行动，触发模型调用熔断。 |

## 7. 结果分析

修复本地放置命令适配后，整体成功率为 92/134，即 68.7%。这说明此前个位数成功率不是模型真实性能，而是工程兼容问题：官方 ReAct prompt 的 `put ... in/on ...` 与当前本地 ALFWorld grammar 的 `move ... to ...` 不一致，导致大量原本接近完成的轨迹在最后一步被环境拒绝。

分任务结果显示，`pick_and_place_simple`、`pick_heat_then_place_in_recep`、`pick_cool_then_place_in_recep` 明显恢复，分别达到 95.8%、87.0%、85.7%。`pick_clean_then_place_in_recep` 为 67.7%，仍有较多清洁工具/目标状态相关的非法动作。`pick_two_obj_and_place` 为 47.1%，主要受多目标记忆和第二个物体搜索影响。`look_at_obj_in_light` 仍只有 11.1%，说明它的瓶颈不在放置命令，而在“先拿目标物体、再使用可用台灯”的任务语义和动作格式。

主要失败模式包括：

1. `look_at_obj_in_light` 中，模型仍会输出 `examine object with desklamp`、`use desklamp on object`、`describe object` 等当前 grammar 不支持或当前状态非法的动作。
2. 部分任务在多次 `Nothing happens.` 后进入 `ouch`、`ouch!`、中文“重新思考”等格式循环。
3. clean/cool 类仍有状态顺序错误，例如没有真正满足清洁/冷却条件，或反复对错误物体执行工具动作。
4. `pick_two_obj_and_place` 需要记住第一个物体是否已经放置、第二个物体在哪里，仍会出现重复探索或中途格式崩坏。

## 8. 本次踩坑与修正记录

第一，50 步口径必须是环境动作数。早期实现容易将每次模型输出都算作一步，导致 `think:` 也消耗步数。当前版本以 `env_steps < max_steps` 控制环境动作上限，并额外设置 `max_agent_turns` 防止模型调用死循环。

第二，parse failure 与 invalid action 必须分开。parse failure 没有发送给环境；invalid action 是发送给环境但被当前动作空间拒绝。当前 `invalid_action_rate` 只用 `action_steps` 做分母。

第三，`max_context_chars` 是字符预算，不是 token 数。本次正式运行没有触发上下文截断，因此低分或失败不能归因于 few-shot 或当前任务被截掉。

第四，也是最关键的一点：要核对 prompt 示例动作和本地 ALFWorld grammar 是否一致。官方 ReAct prompt 里的 `put ... in/on ...` 在当前环境中不一定合法，本地普通放置动作应为 `move ... to ...`。修复这个兼容层后，成功率从 5/134 提升到 92/134。

## 9. 后续改良方向

任务四仍应按单变量消融进行，不能把所有改动混在一起。建议方向如下：

- 针对 `look_at_obj_in_light` 设计任务语义提示，强调先拿目标物体，再使用台灯，并对 `examine/use ... with/on ...` 做格式约束。
- 对连续 `Nothing happens.` 增加显式失败反馈，观察 parse failure loop 是否下降。
- 比较 ReAct、Act-only、CoT-then-Act、JSON/guided decoding，验证显式 thought 是否值得。
- 做 admissible actions 注入实验，预计非法动作率会下降，但这改变了评测协议，必须单独标注。
- 对 clean/cool/heat 和 pick-two 维护简短状态表，记录已拿物品、已完成状态改变、已放置数量。
- 对最好与最差配置做多次运行，报告均值和标准差。

## 10. 交付文件

- `eval.py`
- `prompts/alfworld_3prompts.json`
- `results/task3_react_baseline/trajectories.jsonl`
- `results/task3_react_baseline/summary.json`
- `results/task3_react_baseline/metrics_by_task.csv`
- `requirements-task3.txt`
- `requirements-task3.lock`
- `docs/task3_submission_report.md`
