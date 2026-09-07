# look_at_obj_in_light 灯检任务补跑小报告

## 结论先行

这次补跑说明：原 baseline 在 `look_at_obj_in_light` 上表现差，主要不是 parser 把正确动作解析错了，而是模型在原 prompt 下容易把任务描述 `examine X with desklamp` 直接当成动作输出。ALFWorld 0.4.2 的真实完成协议是先拿目标物体，再在持有目标时执行 `use desklamp`。

官方 ReAct prompt 不是完全错误；它的示例里通常有正确流程。但对当前 Qwen2.5-7B-Instruct 来说，这个协议提示不够显式，压不住任务文字本身的诱导。因此更严谨的表述是：**官方 prompt 对灯检任务协议强调不足，显式 grammar hint 能显著缓解。**

## 专项 18 局消融

这不是替代全量 134 局主结果，而是只看 unseen split 中 18 个灯检任务的诊断实验。

| 实验 | 中文含义 | 相对 baseline 改动 | 成功率 | 结论 |
| --- | --- | --- | ---: | --- |
| `baseline_look_only` | 只跑灯检任务的原 baseline | 只加任务过滤，不改 prompt，不开 repair | 1/18 = 5.6% | 原 baseline 灯检短板明显 |
| `repair_look_only` | 只开动作修正 | 不改 prompt，只尝试把合条件的 `examine X with desklamp` 改成 `use desklamp` | 2/18 = 11.1% | repair 实际触发 0 次，不能证明 parser/action repair 有效 |
| `grammar_hint_look_only` | 只加协议提示 | prompt 加“不要输出 examine-with-lamp，先拿 X，再 use desklamp” | 12/18 = 66.7% | 这是干净的 prompt-only ablation，效果明显 |
| `grammar_hint_plus_repair_look_only` | 协议提示 + 动作修正 | 同时开启 hint 和 repair | 14/18 = 77.8% | 组合上限，不做单变量归因 |

关键判断：D 组效果最好，但它同时改了 prompt 和 action repair，所以不能说“parser 单独修好了问题”。真正清楚的证据是 C 组：只改 prompt 就从 1/18 提到 12/18。

## 全量 134 局复跑

随后把 D 的能力注入方式用于完整 unseen 134 局：保留原 baseline 其他设置，同时开启 `--lookat-lamp-grammar-hint` 和 `--lookat-lamp-action-repair`，不使用任务过滤。

| 指标 | 原任务四 seed0 baseline | baseline + D 式注入 | 变化 |
| --- | ---: | ---: | ---: |
| overall 成功数 | 91/134 | 103/134 | +12 |
| overall 成功率 | 67.9% | 76.9% | +9.0 pp |
| invalid action rate | 30.5% | 28.5% | -2.0 pp |
| parse failure rate | 6.7% | 4.5% | -2.2 pp |
| look-at 成功数 | 2/18 | 14/18 | +12 |
| look-at 成功率 | 11.1% | 77.8% | +66.7 pp |
| look-at repairs | 0 | 2 | +2 |

配对到同一批 game 后，整体有 13 局失败变成功、1 局成功变失败，净 +12；其中 `look_at_obj_in_light` 是 12 局失败变成功、0 局退化。非 look-at 五类合计基本不变，因此全量提升主要来自灯检任务被修复。

## 是否算消融

- `baseline_look_only` vs `repair_look_only`：是 parser/action-repair ablation，但本次 repair 触发 0 次，证据很弱。
- `baseline_look_only` vs `grammar_hint_look_only`：是 prompt-only ablation，证据最清楚。
- `grammar_hint_look_only` vs `grammar_hint_plus_repair_look_only`：是组合上限比较，不用于单变量归因。
- 全量 `baseline + D 式注入`：是增强版 baseline，不应命名为原始 baseline。

## 最终建议

报告中可以说：**加入灯检协议提示与窄 action repair 后，全量成功率从 67.9% 提升到 76.9%，提升主要来自 look-at 子集从 2/18 到 14/18。**

但不要说：**baseline parser 有 bug，修 parser 后成功率提升。**

更准确的结论是：**原 prompt 对 ALFWorld 0.4.2 的灯检动作协议表达不够显式；模型被任务文字诱导输出环境不接受的 examine-with-lamp 动作。显式协议提示解决了主要问题，action repair 只是兜底能力注入。**

## 结果文件

- `results/task4_lookat_lamp_repair/analysis/lookat_before_after.csv`
- `results/task4_lookat_lamp_repair/analysis/lookat_before_after.md`
- `results/task4_lookat_lamp_repair/analysis/full_baseline_d_injection_comparison.csv`
- `results/task4_lookat_lamp_repair/analysis/full_baseline_d_injection_comparison.md`
- `results/task4_lookat_lamp_repair/full_baseline_grammar_hint_plus_repair/summary.json`
- `results/task4_lookat_lamp_repair/full_baseline_grammar_hint_plus_repair/metrics_by_task.csv`
- `results/task4_lookat_lamp_repair/full_baseline_grammar_hint_plus_repair/trajectories.jsonl`
