# look_at_obj_in_light 专项补跑


本节是针对 `eval_out_of_distribution` 中 18 个 unseen `look_at_obj_in_light` 任务的专项诊断，不替代任务三/任务四 134 局主结果。实验继续使用同一 Qwen completion 服务、同一 manifest、`temperature=0`、`max_steps=50`、8 workers，且不使用 `info["admissible_commands"]` 替模型选择动作；该字段只用于统计非法动作。

ALFWorld 0.4.2 的灯检任务完成协议不是 `examine X with desklamp Y`。更可靠的流程是先找到并 `take X from receptacle`，再在持有 X 的状态下到灯处执行 `use desklamp Y`。本次新增两个互相独立的开关：`--task-type-filter look_at_obj_in_light` 只运行该任务类型；`--lookat-lamp-action-repair` 默认为 false，仅在 look-at 任务、模型动作匹配 `examine/look at TARGET with desklamp N`、灯对象是 desklamp、且轨迹中已成功执行过 `take TARGET from ...` 时，将动作改写为 `use desklamp N`。每步 trajectory 记录 `repair_applied`、`repair_type`、`raw_intended_action`、`normalized_before_repair`、`normalized_after_repair`；summary 记录 `task_type_filter`、`action_repairs`、`lookat_lamp_repairs`、`repair_rate`。

需要区分两类修正：`put OBJECT in/on RECEPTACLE -> move OBJECT to RECEPTACLE` 是所有实验共享的本地 grammar compatibility fix；`examine X with desklamp -> use desklamp` 是额外 action repair，必须单独报告。`baseline-look-only vs repair-look-only` 属于 parser/action-repair ablation，不是纯 prompt ablation；`baseline-look-only vs grammar-hint-look-only` 才是 prompt-only ablation；`grammar_hint_plus_repair` 是组合上限，不用于单变量归因。

| experiment | successes | episodes | success_rate | avg_env_steps | invalid_action_rate | parse_failure_rate | lookat_lamp_repairs | avg_total_tokens | wall_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_look_only | 1 | 18 | 0.055556 | 21.277778 | 0.409922 | 0.089385 | 0 | 115458.166667 | 212.694879 |
| repair_look_only | 2 | 18 | 0.111111 | 18.277778 | 0.404255 | 0.089286 | 0 | 100313.277778 | 178.903680 |
| grammar_hint_look_only | 12 | 18 | 0.666667 | 14.222222 | 0.203125 | 0.040380 | 0 | 49754.611111 | 94.410068 |
| grammar_hint_plus_repair_look_only | 14 | 18 | 0.777778 | 14.611111 | 0.258555 | 0.066327 | 2 | 44712.500000 | 92.058259 |

| 对比 | Before | After | Δ Success Rate | Δ Invalid Rate | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline-look-only vs repair-look-only | 1/18 (5.6%) | 2/18 (11.1%) | +5.6 pp | -0.6 pp | parser/action-repair 是否有效 |
| baseline-look-only vs grammar-hint-look-only | 1/18 (5.6%) | 12/18 (66.7%) | +61.1 pp | -20.7 pp | prompt-only 是否有效 |
| grammar-hint-look-only vs grammar-hint+repair | 12/18 (66.7%) | 14/18 (77.8%) | +11.1 pp | +5.5 pp | 组合上限，不做单变量归因 |

## 如何解释这些对照

这四组实验的中文含义如下。

| 实验名 | 中文解释 | 相对 baseline 改了什么 | 能说明什么 |
| --- | --- | --- | --- |
| `baseline_look_only` | 只跑灯检任务的原 baseline | 只加 `--task-type-filter look_at_obj_in_light` 过滤任务，不改 prompt，不开 repair | 原 baseline 在 18 个 unseen 灯检任务上只有 1/18，说明该任务类型是明显短板 |
| `repair_look_only` | 原 baseline + 动作层修正 | 不改 prompt，只开启 `--lookat-lamp-action-repair` | 这不是纯 prompt 实验；本次实际 repair 触发 0 次，所以 1/18 到 2/18 不能证明 repair 本身有效 |
| `grammar_hint_look_only` | 原 baseline + 灯检协议提示 | 不开 repair，只在 prompt 加一句“不要输出 `examine X with desklamp`，先拿 X，再持有 X 时 `use desklamp N`” | 这是干净的 prompt-only 对照；从 1/18 到 12/18，说明明确协议提示非常有效 |
| `grammar_hint_plus_repair_look_only` | 协议提示 + 动作层修正 | 同时加 prompt hint 和 repair | 这是组合上限；成功率 14/18 最高，但不能单独归因给 parser/action repair |

因此，D 效果好不能直接解释为“baseline parser 写错了”。更准确的说法是：模型在原 prompt 下经常把任务文本 `examine the object with the desklamp` 直接仿写成动作 `examine OBJECT with desklamp`，而本地 ALFWorld 0.4.2 的真实完成协议要求先拿目标物体，再执行 `use desklamp`。baseline parser 把模型动作照实送进环境，环境返回 `Nothing happens`；如果评测器把它改成 `use desklamp`，那已经是替模型修动作，属于能力注入，而不是普通解析。

官方 ReAct prompt 也不能简单说“错了”。它的 look-at 示例通常展示了先拿物体再用灯的流程，但这个信息对当前 7B 模型不够强，压不住任务描述本身的诱导。所以报告里更严谨的表述是：**官方 prompt 对 `look_at_obj_in_light` 的动作协议提示不够显式，导致模型常被任务文字诱导出环境不接受的动作；显式 grammar hint 能明显缓解。**

结果文件位于 `results/task4_lookat_lamp_repair/`，每个实验目录包含 `summary.json`、`trajectories.jsonl` 和 `metrics_by_task.csv`；聚合表为 `results/task4_lookat_lamp_repair/analysis/lookat_before_after.csv` 与 `.md`。运行命令如下：

```bash
# A. baseline_look_only
ALFWORLD_DATA=/root/.cache/alfworld python eval.py \
  --gamefile-manifest configs/task4_eval_manifest.json \
  --split eval_out_of_distribution \
  --workers 8 --seed 0 \
  --task-type-filter look_at_obj_in_light \
  --output-dir results/task4_lookat_lamp_repair/baseline_look_only

# B. repair_look_only
ALFWORLD_DATA=/root/.cache/alfworld python eval.py \
  --gamefile-manifest configs/task4_eval_manifest.json \
  --split eval_out_of_distribution \
  --workers 8 --seed 0 \
  --task-type-filter look_at_obj_in_light \
  --lookat-lamp-action-repair \
  --output-dir results/task4_lookat_lamp_repair/repair_look_only

# C. grammar_hint_look_only
ALFWORLD_DATA=/root/.cache/alfworld python eval.py \
  --gamefile-manifest configs/task4_eval_manifest.json \
  --split eval_out_of_distribution \
  --workers 8 --seed 0 \
  --task-type-filter look_at_obj_in_light \
  --lookat-lamp-grammar-hint \
  --output-dir results/task4_lookat_lamp_repair/grammar_hint_look_only

# D. grammar_hint_plus_repair_look_only
ALFWORLD_DATA=/root/.cache/alfworld python eval.py \
  --gamefile-manifest configs/task4_eval_manifest.json \
  --split eval_out_of_distribution \
  --workers 8 --seed 0 \
  --task-type-filter look_at_obj_in_light \
  --lookat-lamp-grammar-hint \
  --lookat-lamp-action-repair \
  --output-dir results/task4_lookat_lamp_repair/grammar_hint_plus_repair_look_only

python scripts/analyze_lookat_lamp_repair.py \
  --results-root results/task4_lookat_lamp_repair
```

轨迹统计显示，baseline look-only 中 `examine/look at ... with desklamp` 出现 36 次；repair-only 中出现 37 次，但因为没有满足“已成功拿着同一目标物体”的触发状态，`lookat_lamp_repairs=0`。因此 repair-only 从 1/18 到 2/18 的差异不能归因为实际 repair 触发，应视为并发推理/轨迹波动下的弱信号。grammar hint 将成功率提升到 12/18，说明主要收益来自 prompt 明确任务协议。组合臂达到 14/18，并触发 2 次 repair，但这两次 `use desklamp` 当步都返回 `Nothing happens`，组合增益不能解释成单独 action repair 的干净效果。

如果 repair 在独立、成对、可复现实验中稳定提升成功率，说明部分失败来自可修复的动作协议；本次结果更接近“提升有限且触发稀疏”，说明瓶颈还包括没有先拿目标物体、找灯失败、对象记忆错误或长程恢复失败。当前建议是保留 `--lookat-lamp-action-repair` 作为单独 ablation/诊断开关，不并入 baseline 默认行为；更值得优先并入的是可解释、prompt-only 的 look-at grammar hint，但它也应作为新 prompt 配置报告，而不是悄悄改写 baseline。

## 全量 134 局 D 式能力注入复跑

为回答“如果把 D 的能力注入方式用于全量 baseline，成功率是否会上升”，额外运行了一次全量 `eval_out_of_distribution` 134 局：保留原 baseline 的 ReAct、匹配 2-shot、完整历史等设置，同时开启 `--lookat-lamp-grammar-hint` 与 `--lookat-lamp-action-repair`，不再加 `--task-type-filter`。输出目录为 `results/task4_lookat_lamp_repair/full_baseline_grammar_hint_plus_repair/`。

| 指标 | 原任务四 seed0 baseline | baseline + D 式注入 | 变化 |
| --- | ---: | ---: | ---: |
| overall 成功数 | 91/134 | 103/134 | +12 |
| overall 成功率 | 67.9% | 76.9% | +9.0 pp |
| overall invalid action rate | 30.5% | 28.5% | -2.0 pp |
| overall parse failure rate | 6.7% | 4.5% | -2.2 pp |
| look-at 成功数 | 2/18 | 14/18 | +12 |
| look-at 成功率 | 11.1% | 77.8% | +66.7 pp |
| look-at repairs | 0 | 2 | +2 |

按同一批 game 配对后，整体有 13 局从失败变成功、1 局从成功变失败，净 +12；其中 `look_at_obj_in_light` 是 12 局失败变成功、0 局退化。非 look-at 五类合计仍为 89/116，说明这次全量提升基本来自灯检任务。

因此，若把这种方式作为“增强版 baseline”全量跑，成功率在本次 seed0 中确实上升：**91/134 -> 103/134**。但报告中不要把它叫作原始 baseline，也不要说它证明 parser 是主因。更准确的表述是：**加入任务特定的灯检协议提示，并保留一个很窄的 action repair 兜底后，显著修复了 look-at 子集，从而提升了全量成功率；主要证据仍指向 prompt 中协议说明不足，而不是 parser-only 问题。**
