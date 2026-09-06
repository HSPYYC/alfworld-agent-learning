# 任务四学习版：Prompt 消融、实现与答辩准备

## 1. 这次真正研究什么

任务一解决模型服务，任务二解决环境，任务三得到一条可审计的 ReAct baseline。任务四不再是“把系统跑通”，而是提出可证伪假设、一次只改一个因素、运行统一评测、解释成功和失败为什么变化。

核心原则是：

```text
先看失败轨迹
  -> 提出假设
  -> 固定公共条件和 game 集
  -> 单变量消融
  -> 同时看成功率、错误率、成本和分任务结果
  -> 对最好/最差补多种子
  -> 区分 prompt 问题与训练问题
```

题目说“至少 4 组”，不是说七条候选各等于一组。一个方向内部可以有多个水平。本项目实现方向 1 至 6，共 11 个唯一配置；Reflexion 会引入失败后的第二次尝试，改变评测预算，故没有和单次尝试实验混合。

## 2. 文件地图

| 文件 | 作用 |
| --- | --- |
| `eval.py` | 统一评测器，支持推理模式、few-shot、合法动作、失败反馈、guided JSON、历史压缩。 |
| `configs/task4_experiments.yaml` | 11 配置矩阵、公共参数、事前假设、排名规则与固定随机示例映射。 |
| `configs/task4_eval_manifest.json` | 固定 134 个 gamefile 及集合 SHA-256。 |
| `configs/task4_failure_annotations.csv` | 正式随机 20 条失败的人工二级标签、证据、建议和置信度。 |
| `scripts/run_task4.py` | dry-run、smoke、seed 0 主实验、最好/最差复现，也支持指定配置补跑 seed 1、2。 |
| `scripts/analyze_task4.py` | 重算一致性、聚合、配对变化、抽样、持久失败和画图。 |
| `tests/test_task4.py` | prompt 选择、JSON、上下文、manifest、单因素矩阵测试。 |
| `requirements-task4*.txt` | ALFWorld 客户端和 vLLM guided decoding 的依赖约束。 |
| `results/task4_prompt_ablation/runs/` | 25 次正式运行的轨迹、summary、CSV、日志和运行元数据。 |
| `results/task4_prompt_ablation/analysis/` | 所有最终表格、随机样本元数据和图表。 |

## 3. 为什么先随机抽 20 条失败

只凭印象挑“有代表性”的失败，容易挑到支持自己观点的案例。这里的正式流程是：

1. 读取任务三 `failure_analysis.csv` 的 42 条失败。
2. 按 `game_id` 排序，消除源文件行顺序变化。
3. 用 Python 标准库 `random.Random(42).sample(population, 20)` 无放回抽样。
4. 保存 seed、算法、样本 ID、源 CSV 和轨迹 JSONL 的 SHA-256。
5. 回到原始 `trajectories.jsonl`，逐步核对 raw output、规范化动作和 observation。
6. 主标签只写最早、最有解释力的根因，终止形式另行统计。

正式样本结果为：协议/格式 8 条，规划/前置条件 6 条，感知/记忆 4 条，探索 2 条。虽然 18/20 最终以 thought 或 parse loop 结束，但“循环”没有成为任何一条的主因。这说明 loop 常是上游错误的症状，例如模型先使用错误 look-at 协议，失败后才开始输出 `ouch`。

之前已有的 `failure_analysis_20_en.csv` 没有删除，因为它仍是有价值的代表性分析；但报告明确说它不是本次固定 seed 的随机样本。

## 4. 公共评测协议为什么要锁死

所有 seed 0 配置都固定：

- Qwen2.5-7B-Instruct，vLLM completion API；
- ALFWorld `eval_out_of_distribution`；
- 同一份 134-game manifest；
- `temperature=0`、`max_tokens=100`；
- 最多 50 个真实环境动作，最多 150 个 agent turns；
- 8 个独立进程，每个进程独立创建 TextWorld env；
- 24000 字符上下文预算；
- 官方动作示例到本地 grammar 的同一规范化层；
- 除正在研究的因素外，其余开关保持 baseline。

manifest 的 SHA-256 是：

```text
9d7b36e410ca9fe4dc1fc5e20cb7750f823e271af775b4f650799e9da3740793
```

为什么还要 seed？`temperature=0` 表示贪心解码，不保证整个并发系统逐 bit 确定。不同 batch 拼接和 GPU 数值路径可能让极接近的 token 发生变化。任务三是 92/134，本次同配置 baseline 是 91/134，正好说明不能假设零温度完全无波动。

## 5. ReAct、Act-only、CoT-then-Act 到底差在哪

### 5.1 ReAct

ReAct 每轮可以在两种输出间选择：

```text
> think: I should inspect the fridge first.
OK.
> go to fridge 1
You arrive at fridge 1...
```

thought 不调用 `env.step`，只是写回历史；动作才推进环境。下一次 thought 能看到新 observation，因此 ReAct 的关键不只是“字更多”，而是推理与反馈交错。

### 5.2 Act-only

Act-only 使用官方文件中的同类 `act_*_1`、`act_*_0` 示例，并明确要求每轮只输出一个环境动作。模型若仍输出 thought，评测器记为 `mode_violation` 和 parse failure，不替模型偷偷删除 thought 后继续。

这能控制行为协议，但不能保证模型服从。实验中 Act-only 的 parse rate 达 29.8%，说明“写一句不要思考”不等于模型必然只行动。

### 5.3 CoT-then-Act

“仓库没有等价 CoT-then-Act prompt”是什么意思？官方 JSON 里有 `react_*` 与 `act_*`，没有 `cot_*`。而交互任务里的 CoT 至少有几种定义：

- 开局一次性写完整计划，再行动；
- 每个动作前思考但不把 thought 放进历史；
- 一次生成计划和全部动作，环境不再反馈；
- 先写自然语言子目标，之后允许或不允许修订。

它们不是同一个实验，所以不能声称有一份天然等价模板。本项目选择第一种，并把操作定义写死：

```text
调用 1：初始 observation -> 一段高层计划
上下文：Act-only 两条示例 + 当前任务 + Initial plan
调用 2...N：每轮只能输出动作
```

计划 prompt 禁止虚构尚未观察到的对象 ID。规划调用、prompt/completion token 和耗时单独记录，不占环境步。

### 5.4 变量究竟是什么

CoT-then-Act 相对 Act-only：

- 相同：`act_*` 示例、only-action 执行协议、环境预算；
- 不同：多一次初始计划调用，后续上下文多一段固定计划。

因此两者差异可解释为“一次性初始规划”的净作用。

ReAct 相对 Act-only/CoT：

- 示例族从 `act_*` 变成 `react_*`；
- 是否允许 thought；
- 推理发生一次还是多次；
- 推理能否看到最新 observation；
- token 与调用数不同。

所以 ReAct 与 Act-only 的差值是一个有意义的“完整策略协议”比较，但不是把同一 prompt 机械删掉 thought 的纯净因果估计。这一点答辩时应主动说明。

### 5.5 结果怎么读

ReAct 67.9%，Act-only 52.2%，CoT-then-Act 50.0%。Act-only 平均 17,996 token，仅为 ReAct 的约 35%，说明显式推理确实有成本，也确实带来总体收益。CoT 没有胜过 Act-only，可能因为不可修订的初始计划在开容器后迅速过时，并产生锚定。

分任务更有信息：Act-only 在 simple place 为 23/24，甚至比 ReAct 的 22/24 好；但 heat 从 20/23 跌到 5/23。显式反馈推理的价值主要出现在多阶段任务，而不是每类任务都同样需要。

## 6. Few-shot 的数量与匹配

### 6.1 四个水平

- 匹配 2-shot：当前任务类型的 `react_PREFIX_1` 和 `_0`。
- 随机非匹配 2-shot：从其他五类共 15 条示例中无放回抽两条，`prompt_seed=42`。
- mixed 6-shot：六类各取 `_1`，顺序固定为项目的 `TASK_ORDER`。
- 0-shot：保留环境说明、当前任务和输出入口，不给轨迹示例。

随机映射既由 seed 算法生成，也显式写入 YAML；单元测试校验两者完全相同。每次 summary 还保存实际 `selected_prompt_keys`，避免只记录“随机 2-shot”却不知道用了哪两条。

### 6.2 为什么 6-shot 不一定更好

上下文学习不是“示例越多越强”。混入不同任务会引入：

- 不相关的动作序列；
- 不同任务的终止协议；
- 更长的固定前缀；
- 更少的历史可用预算；
- 模型对示例中高频模式的错误迁移。

seed 0 结果是匹配 2-shot 67.9% > mixed 6-shot 54.5% > 随机 2-shot 38.1% > 0-shot 6.0%；三种子均值仍是匹配 2-shot 67.41% > mixed 6-shot 54.48% > 随机 2-shot 38.81% > 0-shot 6.22%。6-shot 平均总 token 超过 baseline 两倍，却没有换来更高成功率。这里“匹配”比“数量”更重要。

## 7. 注入 admissible actions

每一回合把当前 `admissible_commands` 排序后追加到 prompt 末尾：

```text
Current admissible actions: ["go to cabinet 1", ...]
If acting, choose exactly one action from this current list.
```

列表不写入永久轨迹；下一状态会重新生成。代码只展示列表，不替模型选动作，也不会把非法输出投影到最近的合法命令。

为什么有争议？列表揭示了当前状态下可执行的对象、容器关系和动作边界。在真实 GUI 中，agent 通常需要先从视觉和工具 schema 推断可操作项；因此该设置比裸文本 observation 更强，报告必须标注。

结果并不像预期那么好：非法动作率只从 30.5% 降到 24.9%，成功率从 67.9% 降到 50.7%，token 几乎翻倍。parse rate 接近零，说明模型会从列表中输出“像动作”的内容；但局部合法不等于推进任务，长列表也可能导致重复和选择偏差。two-object 0/17 是警告：动作约束不能代替多目标状态机。

## 8. 显式失败反馈

只有真实 `env.step` 返回 `Nothing happens.` 时，模型看到：

```text
Nothing happens. The previous action was invalid in the current state.
Do not repeat it; choose a different valid action consistent with location,
inventory, object identity, and container preconditions.
```

本地 parse failure 仍只回传原始 `Nothing happens.`。这样才是单变量：改变环境拒绝后的反馈，不同时改变解析失败机制。

结果是成功率 66.4%，与 baseline 接近；parse-failure loop 23 -> 11，但 50 步/环境结束 7 -> 18，重复动作率反而 21.0% -> 31.1%。提示让模型不那么快“格式崩掉”，却未提供真正的状态恢复策略。错误只是换了一种终止方式时，不能仅凭 parse loop 下降宣布改进。

## 9. Guided JSON 做了什么、没做什么

### 9.1 示例机械转换

官方 ReAct 示例中的模型行：

```text
> think: I need to find the target first.
> go to fridge 1
```

机械转换为：

```json
{"thought":"I need to find the target first.","action":""}
{"thought":"","action":"go to fridge 1"}
```

请求 schema 使用两个完整的 `oneOf` 分支，声明两个字段都存在且恰有一个非空。评测器会再次验证这一条件，因为当前兼容版 xgrammar 不完整执行 `minLength` 等字符串语义关键字。vLLM 请求用 `guided_json` 和 `xgrammar`；动作字符串仍经过原动作规范化，但 schema 不枚举合法动作，所以没有和 admissible-actions 消融混合。

### 9.2 为什么仍有 parse failure

guided decoding 约束正在生成的 token，不会突破 `max_tokens=100`。16 次长 thought 在 JSON 闭合前撞到上限，形成 syntax failure。另 18 次 JSON 是合法的，但 action 写成当前解析器不支持的任务语义，如：

```json
{"thought":"","action":"use cd 1 with desklamp 1"}
```

因此一般 parse rate 是 0.8%，其中纯 JSON 截断约为 16/4268 = 0.37%。guided decoding 解决的是语法形状，不是动作语义、何时结束 thought 或任务策略。

### 9.3 为什么成功率反而下降

JSON 成功率 39.6%，79 局以 thought loop 结束。schema 允许 thought 或 action，但不知道哪一轮必须行动；模型可能持续选择合法 JSON thought。它还把平均 token 推到 81,315。这个实验很好地说明“可解析率”不是“Agent 能力”。

### 9.4 版本坑

当前 vLLM 为 0.6.6.post1。环境中原先的 `xgrammar 0.2.5.post1` 调用了该 vLLM 不具备的 tokenizer 接口，导致 guided decoding 引擎启动失败。独立 vLLM 环境中改为 `xgrammar==0.1.11` 后，JSON 字段结构和正式评测通过。该版本不可靠执行 `minLength`，所以“恰一非空”同时由本地 parser 强制，违规会留下 parse failure，而不是被静默接受。版本约束记录在 `requirements-task4-vllm.txt`；没有改 ALFWorld 客户端的 torch 环境。

## 10. 三种轨迹策略

### 10.1 完整历史

完整保留 trajectory，只有超过 24000 字符才从最旧历史裁剪，few-shot 与当前任务永远保留。baseline seed 0 没有发生字符预算裁剪。

### 10.2 最近 10 turns

无论是否达到预算，只保留最近 10 个“模型输出 + 反馈”块。它能限制 prompt 长度，却会忘记较早访问过的位置、第一个已放置对象或长期子目标。

### 10.3 摘要 + 最近 10 turns

当第 11 个旧块要滑出窗口时，用一次 completion 调用更新结构化记忆，只保留经过 observation 验证的：

- 已访问位置和看到的对象；
- inventory；
- 已完成的 clean/heat/cool 状态；
- 已放置对象 ID；
- 失败动作；
- 剩余子目标。

摘要调用使用 `temperature=0`、`summary_max_tokens=160`，不占环境步，但计入 `llm_calls` 和全部 token。结果为 67.9%，恢复 recent-10 相对 baseline 丢掉的 4.5 pp；平均总 token 40,538，比完整历史低 21.2%，但调用数从 24.75 增至 33.80。

为什么“调用更多、总 token 更少”？每个摘要请求较短，且它让后续每次决策 prompt 都保持紧凑；节省的是重复发送长历史的 prompt token。实际部署仍需考虑请求延迟，不能只看 token。

三种子结果 91/92/87，均值 67.16%，样本标准差 1.97 个百分点。它是预注册 seed 0 排名规则选出的最好配置。后续补跑显示 baseline 为 91/90/90，均值 67.41%，两者基本持平，所以严谨说法是摘要用更少 token 保住了相近表现，不能夸大成显著优于 baseline。

## 11. Completion prompt 和 chat template 到底差在哪

### 11.1 原始 completion 形式

ReAct 文件本质是一段连续文本。模型看到的最后部分类似：

```text
Interact with a household to solve a task. Here are two examples.
[example 1]
[example 2]
Here is the task.
Your task is to: put a hot apple in cabinet.
You are in the middle of a room...
>
```

调用 `/v1/completions` 时，这串字符基本按原样成为 prompt，模型续写一行 `think: ...` 或动作。

### 11.2 “直接套 chat”会发生什么

若代码写成：

```python
client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": whole_react_prompt}],
)
```

Qwen chat template 会在外面加角色 token，概念上变成：

```text
<|im_start|>user
[whole_react_prompt]
<|im_end|>
<|im_start|>assistant
```

原 prompt 中示例的 agent 行只是 user 消息内部的普通文本，不再和当前 assistant 生成处于同一种位置。模型可能改用聊天回答、解释任务、加 `Action:`，而不是按 `> ...` 续写。

若把示例拆成多轮 messages，则每一轮还会插入 `<|im_start|>assistant`、`<|im_end|>` 等边界，序列长度和条件分布再次改变。special token 不是一定“有害”，而是说明两种输入不等价。

### 11.3 正确做 chat 对照的方法

真正研究 chat 格式时，应明确重构：

- system：任务规则和输出协议；
- user：当前 observation；
- assistant：thought/action；
- user 或 tool：环境反馈；
- 每个 few-shot episode 的边界如何表示；
- 是否使用 tool role；
- stop 与解析规则如何对应。

然后把 `prompt_representation=completion/chat` 单独作为一个消融因素。不能一边改角色结构、一边改示例、再把差值叫“chat API 效果”。本任务为了忠实于官方 ReAct prompt，统一使用 `/v1/completions`。

## 12. 指标怎么解释

| 指标 | 定义 | 容易误读的地方 |
| --- | --- | --- |
| success rate | 环境实际给出成功信号的 episode 比例 | 模型自己说“完成了”不算。 |
| avg env steps | 每局 `env.step` 次数均值 | thought 和 parse failure 不算。 |
| avg success env steps | 只在成功局上计算环境动作 | 适合看成功后的效率。 |
| agent turns | 决策模型调用次数 | 包括 thought、action、parse failure。 |
| llm calls | 决策 + 初始规划 + 摘要调用 | CoT/摘要成本必须看它。 |
| invalid action rate | 非当前合法动作 / 已执行动作 | parse failure 不在分母。 |
| parse failure rate | 无法进入当前 thought/action 协议 / agent turns | JSON 合法但 action 内容不可解析也可能算。 |
| repeated action rate | 与上一次执行动作相同 / action steps | 不代表所有循环，如交替动作循环。 |
| total tokens | API usage 的 prompt + completion | 长 prompt 在每轮重复发送，成本增长很快。 |

## 13. 如何亲手运行

### 13.1 环境与服务

推理服务和 ALFWorld 必须分环境：

```bash
cd /root/alfworld-agent
conda activate /root/autodl-tmp/envs/qwen-vllm-clean
python -m pip install -r requirements-task4-vllm.txt
bash scripts/serve.sh
```

另开终端：

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
python -m pip install -r requirements-task4.txt
```

这里默认数据路径仍是 `/root/.cache/alfworld`；软链接细节不影响题目原始命令。

### 13.2 从便宜到昂贵的顺序

```bash
python -m py_compile eval.py scripts/run_task4.py scripts/analyze_task4.py
python -m unittest tests.test_task4
python scripts/run_task4.py --stage dry-run
python scripts/run_task4.py --stage smoke
python scripts/run_task4.py --stage primary
python scripts/run_task4.py --stage replicate
python scripts/analyze_task4.py
```

- dry-run：只打印 11 个命令，不调用模型。
- smoke：每类取一条长轨迹候选，共 6 局，11 配置都跑；摘要配置必须真的触发摘要。
- primary：11 配置各跑 seed 0。
- replicate：读取 seed 0 汇总，默认按 YAML 固定排序挑最好与最差再跑 seed 1、2；也可以用 `--experiments` 指定要补跑的配置。
- analyze：重新读取轨迹，核对 134 个唯一 game、hash、成功数、步数、token 和错误数，再生成表和图。

runner 遇到完整结果会跳过；若目录存在但缺文件，会停止并要求人工检查，不会静默覆盖半成品。

## 14. 结果一致性怎样审计

`scripts/analyze_task4.py` 不信任 summary 中的数字，而是从每行 trajectory 重算：

- episode、success、error 数；
- env/action/agent turn 总数；
- invalid、parse、token 总数；
- 三个比率；
- gamefile 唯一性与集合 hash。

当前 25 次正式运行全部为 134 个唯一 game、六类齐全、hash 相同、`errors=0`，检查结果在 `consistency_checks.csv`。其中 baseline、Act-only、CoT-then-Act、随机非匹配 2-shot、mixed 6-shot、0-shot 和 summary_10 已有三种子；admissible actions、显式失败反馈、guided JSON、recent_10 仍保留 seed 0。按 game ID 与 baseline 配对后，还计算：

```text
baseline 失败 -> variant 成功
baseline 成功 -> variant 失败
两者都成功
两者都失败
```

例如 summary-10 是 11 个新增成功、11 个退化失败，净值 0。这比只看两个相同的 91/134 更能说明策略发生了替换。

## 15. Prompt 能修什么，何时需要训练

Prompt 容易直接影响：

- 输出是自由文本还是 JSON；
- 是否模仿正确动作格式；
- 是否看到合法动作列表；
- 失败后是否收到显式指令；
- 保留多少历史；
- few-shot 中展示哪种任务协议。

Prompt 很难稳定解决：

- 长轨迹中的精确对象 ID 绑定；
- two-object 的计数和先放后拿状态机；
- 未见过 completion protocol 的内化；
- 失败后的真正策略切换，而非换句话重复；
- 区分自我声称完成与环境成功；
- 跨任务泛化而不受错误示例干扰。

11 个 seed 0 配置共同失败 9 局，其中 6 局是 look-at。这是一个有用的停止信号：继续增加自然语言规则可能只增加冲突与 token。更合理的下一步是把正确 `take target -> go lamp -> use lamp` 轨迹、two-object 两遍流程、容器前置条件和错误恢复对构造成训练数据。

## 16. 常见踩坑

1. **把第七项和主表混跑。** Reflexion 是第二次尝试，应单列预算和指标。
2. **把自拟 task protocol hint 当题目消融。** 它可作为后续组合优化，不能挤掉题目候选方向。
3. **把 CoT 当成天然模板。** 必须定义推理发生何时、是否更新、用哪组示例。
4. **用 chat 包住 completion prompt。** role token 改变了条件，必须算新变量。
5. **JSON 请求仍用换行 stop。** 输出会在对象闭合前截断；guided JSON 请求应设 `stop=None`。
6. **认为 schema 能保证任务正确。** 它只约束字段，不理解 ALFWorld 状态。
7. **让 admissible 列表进入永久历史。** 旧状态动作会污染当前选择；本实现每轮动态生成。
8. **把 parse feedback 和 env feedback 一起改。** 会同时改变两个机制，无法归因。
9. **摘要调用不计成本。** 它不占 env step，但必须计入模型调用和 token。
10. **按 worker 位置假定 game 顺序。** TextWorld 内部会重排；应校验 worker 允许集合，最终按 Counter 审计覆盖。
11. **只看 success rate。** 可能把 parse loop 换成步数耗尽，也可能相同总分但成功 episode 完全不同。
12. **版本只写“最新版”。** vLLM 0.6.6.post1 与 xgrammar 0.2.5.post1 实测不兼容，本项目锁为 0.1.11。

## 17. 老师可能追问

**Q：CoT-then-Act 能靠 prompt 控制出来吗？**
能定义和强诱导，但不能保证模型完全服从。因此代码把初始计划拆成独立调用，并把后续 thought 记为 mode violation。控制来自“prompt + 评测器协议 + 日志”，不是一句提示的魔法保证。

**Q：为什么 CoT 比 Act-only 还差？**
计划看不到后续 observation，错误假设无法显式修订；固定计划还会锚定后续动作。它比 Act-only 多 token，不代表信息更有效。

**Q：为什么合法动作列表没有提升成功率？**
合法只表示当前可执行，不表示对目标有用。列表很长、泄漏更强、可能诱发局部循环；它解决动作边界，不解决规划和记忆。

**Q：guided JSON 的 parse rate 为什么不是零？**
16 次被 100-token 上限截断，18 次 JSON 合法但 action 语义无法规范化。约束语法不等于约束内容。

**Q：summary-10 是不是最终最优方案？**
它是 seed 0 预注册排序的最好项，三种子自身较稳定；baseline 补跑后为 91/90/90，均值 67.41%，与 summary_10 的 67.16% 基本持平。严谨结论是它用更少 token 保住了相近成功率，而不是显著优于 baseline。

**Q：为什么错误样本里没有“死循环”主因？**
因为标签取最早根因。18 条虽以循环终止，但通常先有任务协议或状态错误；将终止症状当原因会把 prompt 优化方向带偏。

**Q：为什么不组合最佳开关？**
单变量阶段要回答因果问题。组合可能交互，应该在主效应明确后另开实验，不可拿组合结果冒充某一个因素的收益。

**Q：下一步训练数据怎么做？**
优先覆盖持久失败：look-at 完成协议、two-object 唯一 ID/先放后拿、容器开闭前置条件，以及“错误动作+环境反馈→正确恢复动作”的局部纠错样本。
