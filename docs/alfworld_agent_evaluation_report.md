# ALFWorld LLM Agent 评测与 Prompt 优化实验报告

## 摘要

本文评测本地 `Qwen2.5-7B-Instruct` 在 ALFWorld unseen split 上的文本 Agent 能力，并分析 ReAct prompt 的消融效果。任务三标准 baseline 在 134 局上的成功率为 **68.7% (92/134)**。20 条固定随机失败样本的根因主要是动作格式/任务协议（40%）和规划/前置条件（30%），而不是单纯无法生成文本。任务四在同一 manifest 上比较 11 个配置：任务匹配 2-shot ReAct 达到 **67.9%**，mixed 6-shot 为 **54.5%**，0-shot 仅 **6.0%**。结果说明任务相关示例和动态反馈比示例数量本身更重要；降低 parse rate 或非法动作率也不等价于提高任务成功率。Prompt 可以修补输出协议、短期上下文和局部恢复，但不能稳定替代 7B 模型对状态机、对象绑定和长程恢复策略的学习。

## 1. 推理服务部署与环境验证

### 1.1 服务与环境

| 项目 | 设置 |
|---|---|
| 模型 | Qwen2.5-7B-Instruct |
| 推理后端 | vLLM 0.6.6.post1 |
| API 模型名 | `qwen` |
| 服务接口 | `http://127.0.0.1:8000/v1` |
| GPU | NVIDIA A800 80GB |
| ALFWorld | 0.4.2；TextWorld 1.7.0 |
| unseen split | `eval_out_of_distribution`，134 局 |
| 服务上下文上限 | `--max-model-len 16384` |
| GPU 显存上限 | `--gpu-memory-utilization 0.90` |
| 正式评测并发 | 8 workers |

服务启动命令的关键参数为：

```bash
vllm serve /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
```

任务一压测使用 16 个请求、最大 64 completion tokens：

| 并发 | 墙钟时间 (s) | completion tokens/s | 平均延迟 (s) | P95 延迟 (s) |
|---:|---:|---:|---:|---:|
| 1 | 8.89 | 83.02 | 0.549 | 0.663 |
| 8 | 1.26 | 564.04 | 0.598 | 0.660 |
| 16 | 0.83 | 896.26 | 0.692 | 0.700 |

并发提高后吞吐约为串行的 10.8 倍，平均单请求延迟仅小幅上升，符合 vLLM continuous batching 的行为。任务二环境验证统计 train/seen/unseen 可玩且可解游戏分别为 **3553/140/134**，并完成一次 reset、合法动作执行和人工成功试玩。

### 1.2 Completion 模板与上下文

正式 ReAct 使用 `/v1/completions`，不是把原始 completion 文本直接包进 chat template。官方 prompt 是连续文本，末尾以 `> `等待下一条 `think` 或环境动作；这保留了 ReAct 示例原有的训练/提示位置。Qwen chat 重构会引入角色边界和 special token，若采用必须作为独立协议消融，不能混入 baseline。

评测器对模型生成做以下处理：

- 保留 Few-shot 示例、当前任务和轨迹历史；历史超过 `max_context_chars=24000` 时，从旧到新裁剪，保留最近交互。
- `think:` 不调用环境；只有解析为 action 才执行 `env.step`，`--max-steps 50` 指最多 50 次环境动作。
- 使用统一动作规范化层，将官方示例常见的 `put OBJECT in/on RECEPTACLE` 适配为本地 grammar 的 `move OBJECT to RECEPTACLE`。
- 不把 `admissible_commands` 提供给 baseline 模型，仅用于统计非法动作。

所有正式任务三/四运行均未触发上下文预算截断；因此主要差异应从示例、输出协议、反馈和历史策略解释，而不是意外截断。

## 2. 标准 Baseline 评测结果

任务三 baseline 固定使用：任务类型匹配 2-shot、ReAct、`/v1/completions`、完整历史、`temperature=0`、`max_tokens=100`、最大 50 个环境动作、unseen 134 局、8 workers。

### 2.1 总体与分任务结果

| Task type | Episodes | Success | Success rate | Avg Env Steps | Parse Failure Rate | Invalid Action Rate |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | **134** | **92** | **68.7%** | **17.01** | **5.2%** | **32.2%** |
| `pick_and_place_simple` | 24 | 23 | 95.8% | 15.17 | 1.9% | 8.8% |
| `pick_clean_then_place_in_recep` | 31 | 21 | 67.7% | 19.77 | 4.3% | 50.4% |
| `pick_heat_then_place_in_recep` | 23 | 20 | 87.0% | 13.52 | 1.8% | 16.7% |
| `pick_cool_then_place_in_recep` | 21 | 18 | 85.7% | 16.33 | 0.0% | 48.1% |
| `look_at_obj_in_light` | 18 | 2 | 11.1% | 19.22 | 7.6% | 39.9% |
| `pick_two_obj_and_place` | 17 | 8 | 47.1% | 17.82 | 13.1% | 12.5% |

baseline 能稳定处理简单取放和大部分 heat/cool 流程，但 `look_at_obj_in_light` 与 two-object 暴露出明显的任务状态机和对象记忆瓶颈。clean/cool 的非法动作率也较高，说明“动作语法合法”不能替代“动作满足前置条件”。

### 2.2 终止原因

| Termination reason | Episodes | 占比 | 含义 |
|---|---:|---:|---|
| `success` | 92 | 68.7% | 环境返回成功奖励并结束 |
| `env_done_without_success` | 8 | 6.0% | 环境结束或达到动作上限但未完成 |
| `parse_failure_loop` | 18 | 13.4% | 连续输出无法规范化为 thought/action |
| `thought_loop` | 16 | 11.9% | 连续只输出 thought，未推进环境 |

这里需要区分两个概念：正式 baseline 的总体 parse failure rate 是 **5.2%**，不是 60% 以上；“循环终止”是 episode 级终止原因，可能由少量较长连续错误触发。把终止原因占比直接当成逐 turn parse rate 会夸大格式错误的作用。

## 3. 失败轨迹归因分析

从任务三的 42 条失败中按 `game_id` 排序，再以 `random.Random(42).sample(..., 20)` 无放回抽样。每条轨迹按最早且最有解释力的根因标注，终止形式单独统计。

### 3.1 根因分布

| 根因 | 数量 | 占比 | 文本分布 |
|---|---:|---:|---|
| 动作格式/任务协议 | 8 | 40% | `████████` |
| 规划/前置条件 | 6 | 30% | `██████` |
| 感知/记忆 | 4 | 20% | `████` |
| 探索/找物 | 2 | 10% | `██` |
| 死循环/恢复 | 0 | 0% | |

抽样失败的终止形式为：thought loop 11 条、parse-failure loop 7 条、环境结束/步数耗尽 2 条。循环通常是上游错误的症状，例如模型先使用错误的 `look_at_obj_in_light` 动作协议，之后才开始输出 `ouch` 或重复 thought。

### 3.2 主要失败模式

1. **Look-at 任务协议错误。** 模型常输出 `examine object with desklamp`、`use desklamp on object` 或 `describe object`。本地可行流程通常要求先找到并拿起目标物体，再到灯处执行 `use desklamp`。
2. **状态前置条件错误。** clean/heat/cool 任务中，模型可能尚未拿到物体就调用工具，或处理完成前直接放置；动作形式看似合法，但环境状态不满足。
3. **对象身份与长程记忆错误。** two-object 任务需要同时记住第一个物体是否已放置、第二个物体的 ID 和位置，模型会重复计数或回到已检查容器。
4. **恢复策略不足。** `Nothing happens.` 之后，模型没有形成“记录失败动作、排除该动作、重新检查位置/库存”的闭环，而是重复相同动作或生成 `ouch`。

因此，裸 Prompt 的主要问题不是“模型完全不会规划”，而是规划无法稳定落到本地动作协议、环境前置条件和可修订的状态记录上。Prompt 可改善这些行为的可见表达，但不能保证 7B 模型在长轨迹中持续维护正确状态。

## 4. Prompt 消融实验与假设验证

所有 seed 0 配置使用同一 134-game manifest、同一 Qwen 服务、completion 接口、历史策略、动作规范化、解码参数和步数上限；正式 few-shot 四组只改变示例选择策略。

### 4.1 总体结果

| 配置 | Success | Success Rate | Invalid Action Rate | Parse Failure Rate | Avg Total Tokens |
|---|---:|---:|---:|---:|---:|
| **Baseline：匹配 2-shot ReAct** | **91/134** | **67.9%** | 30.5% | 6.7% | 51,440 |
| Act-only | 70/134 | 52.2% | 12.7% | 29.8% | 17,996 |
| CoT-then-Act | 67/134 | 50.0% | 25.0% | 24.2% | 30,037 |
| 随机非匹配 2-shot | 51/134 | 38.1% | 31.1% | 10.6% | 75,566 |
| mixed 6-shot | 73/134 | 54.5% | 15.7% | 15.8% | 105,556 |
| 0-shot | 8/134 | 6.0% | 58.1% | 35.1% | 16,196 |
| 注入合法动作 | 68/134 | 50.7% | 24.9% | 0.1% | 97,596 |
| 显式失败反馈 | 89/134 | 66.4% | 39.0% | 4.3% | 66,562 |
| Guided JSON | 53/134 | 39.6% | 9.0% | 0.8% | 81,315 |
| 最近 10 turns | 85/134 | 63.4% | 20.9% | 9.5% | 42,819 |
| 摘要 + 最近 10 | 91/134 | 67.9% | 19.3% | 10.1% | 40,538 |

### 4.2 逐变量分析

#### 推理策略：ReAct / Act-only / CoT-then-Act

- **事前假设：** ReAct 最好；Act-only 最省 token；一次性 CoT 介于二者之间。
- **实验结果：** ReAct 67.9%，Act-only 52.2%，CoT-then-Act 50.0%；Act-only 总 token 仅为 ReAct 的 35.0%。
- **归因与边界：** 动态 ReAct 能在新 observation 后修订计划，优势主要出现在 heat、clean 等多阶段任务。Act-only 虽减少非法动作和 token，却更容易违反 only-action 协议；CoT 的一次性计划无法看到后续容器内容，错误计划会产生锚定。假设中“成本”和“ReAct 优势”成立，“CoT 介于二者之间”被推翻。

#### Few-shot：匹配、随机、mixed6、0-shot

- **事前假设：** 匹配示例优于随机和 0-shot；6-shot 不保证单调提升。
- **实验结果：** 匹配 2-shot 67.9%，mixed6 54.5%，随机非匹配 2-shot 38.1%，0-shot 6.0%。mixed6 的非法动作率降至 15.7%，但 parse failure 升至 15.8%，平均 token 达 105,556，约为 baseline 的 2.05 倍。
- **归因与边界：** 任务相关性比数量更关键。mixed6 用一条同类和五条异类轨迹，移除了 baseline 的第二条同类示例；异类任务还会竞争动作协议和注意力。当前四臂同时改变了数量与匹配方式，不能严格定位饱和点。必须强调：这些正式结果都在同一 completion 执行协议下有效；参考 chat 重构探针回答的是不同协议，不能替换主表。

#### 动作空间：是否注入 admissible actions

- **事前假设：** 合法动作列表会显著降低非法动作并提高成功率，但增加 token。
- **实验结果：** 非法动作率由 30.5% 降至 24.9%，parse failure 几乎归零，但成功率由 67.9% 降至 50.7%，总 token 增至 97,596。
- **归因与边界：** 候选列表改善了输出外壳，却把模型推向“局部合法、全局不推进”的动作；它还泄露当前状态下可执行对象和容器关系，属于比裸 ReAct 更强的信息条件。动作合法性不是任务完成能力。

#### 失败反馈：原始反馈 vs 显式恢复提示

- **事前假设：** 显式反馈会减少重复非法动作和循环终止，对规划错误帮助有限。
- **实验结果：** 成功率 66.4%，parse failure 由 6.7% 降至 4.3%，但非法动作率由 30.5% 升至 39.0%，重复动作率由 21.0% 升至 31.1%。
- **归因与边界：** 提示改变了终止形式，使部分 parse loop 变成长时间无效探索；它没有给模型补上对象状态表或恢复策略。假设只部分成立，不能用 parse rate 单项指标宣布改进。

#### 输出结构：自由文本 vs Guided JSON

- **事前假设：** JSON 能消除语法错误，但未必减少语义非法动作，约束可能损害推理。
- **实验结果：** parse failure 由 6.7% 降至 0.8%，非法动作率降至 9.0%，但成功率只有 39.6%，平均 token 81,315，79 局以 thought loop 结束。
- **归因与边界：** guided decoding 约束的是 JSON 外壳，不知道当前轮必须行动，也不判断动作是否满足任务状态。长 thought 在 100-token 上限处截断会造成 JSON 不闭合；合法 JSON 仍可能包含不支持的动作语义。格式正确不等于策略正确。

#### 历史策略：完整 / 最近 10 / 摘要 + 最近 10

- **事前假设：** 完整历史最好；最近窗口最省但损害长程跟踪；摘要可恢复部分能力但增加调用和记忆失真风险。
- **实验结果：** 最近 10 为 63.4%，摘要 + 最近 10 与 baseline 同为 67.9%；摘要方案总 token 降至 40,538，但平均 LLM calls 增至 33.80。
- **归因与边界：** 摘要在成本上有效，并恢复了部分窗口损失；但 seed 0 的平局不足以证明它总体优于完整历史。摘要可能丢失对象 ID、容器状态和失败事实，必须结合跨 seed 和逐步记忆准确性评估。

### 4.3 Prompt 能解决什么，不能解决什么

| 问题 | Prompt 可修补程度 | 证据 |
|---|---|---|
| `Thought/Action` 外壳与动作前缀 | 较高 | Guided JSON 将 parse rate 降至 0.8%，但成功率未同步提升 |
| 官方示例与本地 grammar 的表达差异 | 高 | 统一 `put -> move` 适配后 baseline 达 68.7% |
| 短期失败反馈和局部重复动作 | 中等 | 显式反馈降低 parse failure，但重复/非法动作反而上升 |
| 任务类型的标准步骤 | 中等 | 匹配 2-shot 明显优于随机/0-shot |
| 长程对象 ID、库存和容器状态 | 较低 | two-object、clean、look-at 仍是主要失败类型 |
| 失败后的策略切换与真实状态机 | 较低 | `Nothing happens.` 后持续 `ouch` 或重复动作 |

因此，Prompt 优化适合约束协议、提供局部示范和压缩上下文，不应被解释为对模型策略能力的替代训练。若错误跨多个 prompt 变体持续出现，应将失败前缀、正确恢复动作和环境状态转移构造成 SFT/DPO/STaR 数据。

## look_at_obj_in_light 专项补跑

本节是针对 `eval_out_of_distribution` 中 18 个 unseen `look_at_obj_in_light` 任务的专项诊断，不替代 134 局主结果。新增 `--task-type-filter look_at_obj_in_light` 后，四个 seed0 子集实验均只运行这 18 局，并继续禁止用 `admissible_commands` 替模型选动作。

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

归因边界如下：`baseline-look-only vs repair-look-only` 是 parser/action-repair ablation，不是纯 prompt ablation；`baseline-look-only vs grammar-hint-look-only` 才是 prompt-only ablation；`grammar_hint_plus_repair` 是组合上限，不用于单变量归因。`put -> move` 是所有实验共享的 grammar compatibility fix，而 `examine X with desklamp -> use desklamp` 是额外 action repair，需要单独报告。

这意味着 D 组效果好不能直接说明 baseline parser 写错。baseline parser 是把模型给出的 `examine OBJECT with desklamp` 照实执行；环境不接受，是因为 ALFWorld 0.4.2 的灯检完成协议要求先拿目标物体，再 `use desklamp`。官方 ReAct 示例并非完全错误，但对这个协议强调不足，当前 7B 模型容易被任务文字诱导成错误动作；显式 grammar hint 才是本次最清楚的有效因素。

本次 repair-only 虽从 1/18 到 2/18，但 `lookat_lamp_repairs=0`，因此没有直接证据表明 action repair 本身造成提升。prompt-only grammar hint 提升到 12/18，说明许多失败来自可提示的灯检协议；组合臂 14/18 且触发 2 次 repair，但不能作单变量归因。若未来 repair 稳定提升，说明部分失败来自可修复动作协议；若提升有限，则瓶颈还包括没有先拿目标物体、找灯失败、对象记忆错误或长程恢复失败。当前建议将 repair 保留为单独 ablation 开关，不并入 baseline 默认行为。

将 D 式能力注入用于全量 134 局后，baseline 从 91/134 提升到 103/134，成功率从 67.9% 到 76.9%。配对结果为 13 局失败变成功、1 局成功变失败，净 +12；其中 look-at 从 2/18 到 14/18，贡献全部净增。这个结果说明全量成功率确实会上升，但应报告为“baseline + look-at 协议提示/repair 的增强配置”，不是原始 baseline，也不能归因为 parser-only。

## 5. 开放性问题

### 5.1 ALFWorld 与真实 GUI/工具调用的差距

ALFWorld 提供结构化文本 observation 和离散动作，缺少像素定位、遮挡、点击坐标、异步延迟、权限错误、网络波动与不可逆副作用。它适合评估任务分解、对象状态跟踪、工具前置条件和失败恢复，不能直接代表视觉 grounding 或真实工具可靠性。可迁移的是“根据反馈修订计划”的抽象能力；不可直接迁移的是从视觉和动态界面中发现目标、判断可点击区域及处理非结构化故障的能力。报告中的成功率应被理解为文本闭环上限，而非 GUI Agent 的端到端成功率。

### 5.2 除成功率外如何评价探索效率与策略消耗比

应同时记录成功条件下的环境步数、首次发现目标步数、访问过的地点数、重复访问率、无效动作率、失败恢复次数和相对最短可行路径的 regret。成本方面可定义策略消耗比：`成功任务数 / (环境动作数 + λ·LLM调用数 + μ·总token/1000)`，其中 λ、μ 需在实验前固定，避免事后调权重。还可报告“有效动作率”“每个成功子目标的 token”与“完成进度分数”，将已找到物体、已完成清洁/加热、已放置数量作为中间状态，避免把所有未成功 episode 都视为同一种失败。

### 5.3 何时停止调 Prompt，转向数据与微调

当多个 prompt 只改变错误外观、收益小于跨 seed 波动，或相同任务持续因状态转移、对象绑定和恢复策略失败时，应停止继续堆提示。若增加示例只带来更长上下文、更高 token 和新的协议冲突，却没有稳定提升成功率，说明瓶颈已从条件表达转为模型策略能力。此时应冻结评测协议，把成功轨迹提炼为正确状态转移，把失败轨迹切成“错误前缀—正确下一步/恢复”样本，先做 SFT，再用成败或步骤质量构造 DPO 偏好；STaR 可用于迭代生成和筛选推理轨迹。

### 5.4 如何利用 1000 条成功/失败轨迹提升模型

先按任务类型、成功状态、错误根因和轨迹长度分层去重，留出按 gamefile 隔离的验证集，防止同布局泄漏。成功轨迹用于学习正确动作协议、前置条件和简短 thought；失败轨迹不应整段模仿，而应标出首次错误、环境反馈和可验证的恢复动作。SFT 学习格式与基本状态机，DPO 用“正确恢复轨迹优于重复错误轨迹”的成对样本，STaR 则让模型生成候选推理后只保留环境验证成功的轨迹。训练后必须用未见 game 和冻结 completion 协议复测，同时报告成功率、invalid action、parse rate、token 和恢复成功率。

## 6. 交付与复现

核心交付文件如下：

- 任务一：`scripts/serve.sh`、`scripts/check_llm.py`、`scripts/benchmark_llm.py`、`requirements-task1*.txt`、`logs/benchmark_task1_generation.json`。
- 任务二：`env_check.py`、`scripts/setup_task2_env.sh`、`configs/base_config.yaml`、`logs/env_check_task2.log`、`logs/manual_play_task2_clean.log`。
- 任务三：`eval.py`、`prompts/alfworld_3prompts.json`、`results/task3_react_baseline/`、`docs/task3_submission_report.md`。
- 任务四：`configs/task4_experiments.yaml`、`configs/task4_eval_manifest.json`、`scripts/run_task4.py`、`scripts/analyze_task4.py`、`scripts/analyze_lookat_lamp_repair.py`、`tests/test_task4.py`、`results/task4_prompt_ablation/`、`results/task4_lookat_lamp_repair/`、`docs/task4_submission_report.md`、`docs/lookat_lamp_repair_report.md`。
- 本总报告：`docs/alfworld_agent_evaluation_report.md`。

复现顺序：启动 vLLM，激活 `alfworld` 环境，运行 `python -m unittest tests.test_task4`，执行 `scripts/run_task4.py --stage dry-run`、`--stage smoke`、`--stage primary`，最后运行 `scripts/analyze_task4.py`。主结果使用固定 unseen manifest SHA-256：




