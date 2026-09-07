# look_at_obj_in_light 专项补跑

| experiment | successes | episodes | success_rate | avg_env_steps | invalid_action_rate | parse_failure_rate | lookat_lamp_repairs | avg_total_tokens | wall_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_look_only | 1 | 18 | 0.055556 | 21.277778 | 0.409922 | 0.089385 | 0 | 115458.166667 | 212.694879 |
| repair_look_only | 2 | 18 | 0.111111 | 18.277778 | 0.404255 | 0.089286 | 0 | 100313.277778 | 178.903680 |
| grammar_hint_look_only | 12 | 18 | 0.666667 | 14.222222 | 0.203125 | 0.040380 | 0 | 49754.611111 | 94.410068 |
| grammar_hint_plus_repair_look_only | 14 | 18 | 0.777778 | 14.611111 | 0.258555 | 0.066327 | 2 | 44712.500000 | 92.058259 |

## 前后对比

| 对比 | Before | After | Delta Success Rate | Delta Invalid Rate | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline-look-only vs repair-look-only | 1/18 (5.6%) | 2/18 (11.1%) | +5.6 pp | -0.6 pp | parser/action-repair 是否有效 |
| baseline-look-only vs grammar-hint-look-only | 1/18 (5.6%) | 12/18 (66.7%) | +61.1 pp | -20.7 pp | prompt-only 是否有效 |
| grammar-hint-look-only vs grammar-hint+repair | 12/18 (66.7%) | 14/18 (77.8%) | +11.1 pp | +5.5 pp | 组合上限，不做单变量归因 |
