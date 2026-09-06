# 任务二学习版说明：ALFWorld 文本环境部署与理解

## 1. 任务二的真正目的

任务二不是让模型完成任务，也不是评测成功率。它的真正目的，是确认后续 Agent 实验需要的“环境端”已经可用。

整个大作业的链路可以拆成四段：

```text
LLM 推理服务 -> ALFWorld 环境 -> Agent 循环 -> 实验记录与分析
```

任务一已经解决了第一段：本地有一个 OpenAI-compatible 的 Qwen 推理服务。任务二解决第二段：本地 Python 程序和人类都能进入 ALFWorld 文本环境，看到任务、输入动作、获得反馈。

如果任务二没有打通，任务三的 ReAct 循环就没地方交互。模型可以生成动作，但 Python 程序无法 `env.reset()` 或 `env.step()`，实验就跑不起来。所以任务二的交付物看起来简单，其实是在给后续评测打地基。

## 2. 为什么要交这些东西

任务二要求交付：

1. `env_check.py`
2. 一张手动玩通一局的完整交互 log

我额外保留了配置、依赖和程序化验证日志，是为了让整个仓库可复现。

| 文件 | 为什么要有 | 汇报时怎么说 |
| --- | --- | --- |
| `configs/base_config.yaml` | 告诉 ALFWorld 数据在哪里、用哪个环境类、启用哪些任务类型。 | “这是环境初始化配置，后续所有 reset/step 都依赖它。” |
| `env_check.py` | 用 Python 代码验证 ALFWorld 能否被程序调用。 | “它是最小 smoke test，证明后续 Agent loop 可以接管环境。” |
| `logs/env_check_task2.log` | 保存 `env_check.py` 的真实运行输出。 | “不是只写了脚本，还保存了实际跑通证据。” |
| `logs/manual_play_task2_clean.log` | 保存人工试玩的完整可读交互记录。 | “这是提交用 log，已经去掉终端控制符和服务器登录提示，只保留和游戏过程有关的内容。” |
| `requirements-task2.txt` | 记录任务二直接依赖。 | “复现时先装这些关键包。” |
| `requirements-task2.lock` | 记录当前完整 Python 包版本。 | “如果复现失败，可以对照锁定版本排查。” |
| `scripts/setup_task2_env.sh` | 自动创建/复用 conda 环境并安装依赖。 | “这是可复现部署入口。” |

老师如果问“为什么不是只交 `alfworld-play-tw` 的截图或 log”，可以这样答：

> 手动 log 只能证明人能玩，不能证明程序能接管环境。后续任务三需要 LLM 自动玩 ALFWorld，所以必须提前验证 Python API：`reset()` 能返回观察，`info` 能返回合法动作，`step()` 能接受动作并返回下一状态。

老师如果问“为什么 `env_check.py` 只跑一步，不直接跑完整任务”，可以这样答：

> 任务二的目标是部署和接口验证，不是策略评测。完整完成任务属于任务三的 Agent loop。这里跑一步就足够验证环境初始化、任务读取、合法动作列表、动作执行和反馈返回都正常。

## 3. `alfworld-play-tw` 和 `env_check.py` 的区别

这两个东西都在“玩 ALFWorld”，但验证角度不同。

| 方式 | 谁在玩 | 主要验证什么 | 为什么需要 |
| --- | --- | --- | --- |
| `alfworld-play-tw` | 人 | 人能否看懂任务、输入动作、理解环境反馈。 | 让你建立直觉，知道 benchmark 长什么样。 |
| `env_check.py` | Python 程序 | 代码能否加载环境、reset、读取 info、step。 | 为后续 LLM Agent 自动评测做接口验证。 |

可以把它想成两种检查：

```text
人类试玩：我知道这个游戏怎么玩。
程序验证：我的代码知道怎么打开这个游戏、往里发动作、拿回反馈。
```

任务三会把“人输入动作”的位置替换成“LLM 生成动作”：

```python
obs, info = env.reset()
action = llm(prompt)
obs, reward, done, info = env.step([action])
```

所以 `env_check.py` 是任务三的前置测试，不是多余文件。

## 4. 为什么要新建 `alfworld` 环境

ALFWorld 依赖 TextWorld、Jericho、spaCy 等包，和 vLLM/torch 的依赖栈完全不同。文档里也提示了环境可能冲突，所以任务二单独建：

```bash
conda create -n alfworld python=3.10 -y
```

本机直接使用默认镜像时会被代理/国内镜像组合拖慢或触发 SSL EOF，所以实际采用：

```bash
conda create -y --override-channels -c defaults -n alfworld python=3.10
```

这不是改变任务要求，而是绕开 `.condarc` 中的国内镜像；环境名仍然是 `alfworld`，Python 版本仍然是 3.10。Task 1 的 vLLM 环境 `/root/autodl-tmp/envs/qwen-vllm-clean` 没有被改动。

汇报时可以说：

> 任务一和任务二的依赖冲突风险不同。任务一需要 vLLM 和 CUDA/torch 栈，任务二需要 TextWorld/Jericho/spaCy 栈。为了避免互相污染，我按要求为 ALFWorld 单独建了 `alfworld` 环境。

## 5. 为什么只装 text-only ALFWorld

文档说“纯文本环境足够”，所以安装：

```bash
pip install alfworld
pip install PyYAML
```

没有安装 `alfworld[full]`，因为 full/vis 会引入 THOR 视觉环境、Mask R-CNN 等额外依赖。本练习的任务二、任务三和任务四都围绕 TextWorld 文本交互，不需要图像观察。

这点汇报时要说清楚：

> 本练习目标是跑通文本版 ALFWorld 上的 LLM Agent，不涉及视觉导航，因此只安装 text-only 环境。这样依赖更轻，复现风险更低，也符合任务说明。

## 6. 数据放在哪里，为什么不用仓库目录

文档示例是：

```bash
export ALFWORLD_DATA=~/.cache/alfworld
alfworld-download
```

本机实际使用：

```bash
export ALFWORLD_DATA=/root/.cache/alfworld
alfworld-download
```

原因是 `~/.cache/alfworld` 是 ALFWorld 官方默认数据目录，直接使用这个路径可以保证文档里的命令原样可用。数据不放进仓库，避免仓库变得很大。

你可以这样理解：仓库存代码、配置、报告和 log；数据集放外部数据盘。这样才是干净的实验仓库。

## 7. `base_config.yaml` 是哪里来的，为什么必须有

ALFWorld 初始化环境时不是只需要一句 `AlfredTWEnv()`。它还需要知道：

- 数据集路径在哪里。
- 当前跑 train、seen 还是 unseen。
- 用文本环境还是视觉环境。
- 启用哪几类任务。
- PDDL domain 和 TextWorld grammar 文件在哪里。
- 每局最多允许多少步。

这些信息集中放在 `configs/base_config.yaml`。

当前项目里的关键字段是：

```yaml
general:
  training_method: dqn

env:
  type: AlfredTWEnv
  task_types: [1, 2, 3, 4, 5, 6]

dataset:
  data_path: $ALFWORLD_DATA/json_2.1.1/train
  eval_id_data_path: $ALFWORLD_DATA/json_2.1.1/valid_seen
  eval_ood_data_path: $ALFWORLD_DATA/json_2.1.1/valid_unseen

logic:
  domain: $ALFWORLD_DATA/logic/alfred.pddl
  grammar: $ALFWORLD_DATA/logic/alfred.twl2
```

任务文档说 `base_config.yaml` 在 ALFWorld 包的 `configs/` 目录中。当前 pip 安装的 `alfworld==0.4.2` 没有把这个 yaml 放进 site-packages，但官方 GitHub 仓库有完整配置。项目里的版本是按官方配置和当前包源码所需字段整理出的最小文本环境配置。

如果老师问“为什么不是直接复制包里的 config”，可以回答：

> 当前 pip 包安装后没有随包提供 `configs/base_config.yaml`，所以我参考官方仓库和 `AlfredTWEnv` 源码，保留文本环境初始化所需字段，放到项目的 `configs/base_config.yaml` 中。脚本已经用它成功初始化 unseen 环境。

## 8. `env_check.py` 每一段在做什么

`env_check.py` 是最小程序化验证脚本，核心不是“赢一局”，而是验证环境 API 可用。

### 8.1 处理数据目录

```python
data_dir = ensure_data_dir(args.data_dir)
os.environ["ALFWORLD_DATA"] = resolved
```

作用：确保程序能找到 ALFWorld 数据。优先顺序是：命令行传入 `--data-dir`、已有环境变量、本机默认数据路径 `/root/.cache/alfworld`、最后才是 `~/.cache/alfworld`。

这样写的原因是避免你忘记 `export ALFWORLD_DATA=...` 后脚本直接失败。

### 8.2 读取配置

```python
config = load_config(Path(args.config))
```

作用：把 `configs/base_config.yaml` 读成 Python dict。后续初始化环境时，ALFWorld 会从里面读取 `env.type`、`dataset.*`、`logic.*` 等字段。

### 8.3 统计数据集规模

```python
print_dataset_summary(config)
```

作用：统计 train、seen、unseen 中“可玩且可解”的 game 数量，并按六类任务拆分。

为什么要统计？因为文档里明确问：

> `eval_in_distribution`（seen）和 `eval_out_of_distribution`（unseen）分别是多少局？区别在哪？

脚本不是只口头回答，而是用本地数据实际算出来。

### 8.4 初始化环境

```python
env_class = get_env_class(config["env"]["type"])
env = env_class(config, train_eval=split)
env = env.init_env(batch_size=batch_size)
```

作用：创建 ALFWorld 文本环境。这里 `split` 默认是 `eval_out_of_distribution`，也就是后续任务三要跑的 unseen。

文档示例里用：

```python
env = getattr(environment, config["env"]["type"])(...)
```

当前安装版本里 `alfworld.agents.environment` 没有直接导出 `AlfredTWEnv`，但提供了 `get_environment()`。所以脚本先尝试 `getattr`，失败后回退到 `get_environment()`，兼容当前版本。

### 8.5 reset 一局任务

```python
obs, info = env.reset()
```

作用：让环境打开一局 game。`obs[0]` 是文本观察，里面有房间描述和任务目标；`info` 是额外信息，包括当前合法动作。

实际输出里有：

```text
Your task is to: put a mug in desk.
```

这说明环境已经能读取具体任务。

### 8.6 打印合法动作

```python
commands = info["admissible_commands"][0]
```

作用：读取当前状态下环境允许的动作。例如：

```text
go to bed 1
go to desk 1
go to drawer 1
```

这对任务二很重要，因为 ALFWorld 的动作不是随便写自然语言。动作需要符合 TextWorld grammar，比如 `go to ...`、`take ... from ...`、`put ... in/on ...`。

### 8.7 执行一步动作

```python
action = pick_safe_action(commands)
obs, reward, done, info = env.step([action])
```

作用：确认环境真的能接收动作并返回下一步观察。

为什么不用文档示例里的 `go to countertop 1`？因为那只是例子，不保证每个任务都有 `countertop 1`。本次 reset 到卧室任务，合法动作里有 `go to bed 1`、`go to desk 1` 等，没有 countertop。所以脚本从 `admissible_commands` 中选一个真实合法的导航动作，避免验证脚本因为随机任务不同而失败。

汇报时可以这样说：

> 文档里的 `go to countertop 1` 是示例动作。为了让脚本对不同初始任务都稳健，我从环境返回的 admissible commands 中选择一个合法动作执行。这样验证的是环境 API，而不是某个固定房间里是否存在 countertop。

## 9. `env_check_task2.log` 证明了什么

`logs/env_check_task2.log` 保存了脚本真实输出。它证明：

- `alfworld==0.4.2` 和 `textworld==1.7.0` 可导入。
- `ALFWORLD_DATA` 指向 `/root/.cache/alfworld`。
- train/seen/unseen 数据都能被扫描。
- unseen split 初始化出 134 个可玩可解 games。
- 环境成功 reset 到一局具体任务。
- 环境返回了 `admissible_commands`。
- 对 `go to bed 1` 执行 `step()` 后得到了下一步观察。

这份 log 是“环境真的跑过”的证据，不是配置说明。

## 10. 已测到的数据规模

| Split | 可玩且可解游戏数 | 用途 |
| --- | ---: | --- |
| train | 3553 | 训练/示例来源 |
| eval_in_distribution / valid_seen | 140 | seen 验证；场景分布与训练更接近 |
| eval_out_of_distribution / valid_unseen | 134 | unseen 验证；后续主要评测集 |

unseen 的六类任务数量分别是：

| 任务类型 | 数量 | 难点 |
| --- | ---: | --- |
| `pick_and_place_simple` | 24 | 找物体、找目标容器、放置 |
| `pick_clean_then_place_in_recep` | 31 | 多一步清洗，需要找到 sinkbasin |
| `pick_heat_then_place_in_recep` | 23 | 多一步加热，需要找到 microwave，顺序容易错 |
| `pick_cool_then_place_in_recep` | 21 | 多一步冷却，需要找到 fridge，常涉及开关容器 |
| `look_at_obj_in_light` | 18 | 找物体后还要靠近/使用灯光类目标 |
| `pick_two_obj_and_place` | 17 | 要找两个同类物体，记忆和循环控制更难 |

这里的“可玩且可解”不是简单数 `traj_data.json`，而是按照 ALFWorld 源码过滤：排除 movable receptacle、Sliced 任务、缺少 `game.tw-pddl` 的任务，以及标记为不可解的任务。

## 11. seen 和 unseen 的区别

`eval_in_distribution` 对应数据目录 `valid_seen`，本地统计为 140 局。

`eval_out_of_distribution` 对应数据目录 `valid_unseen`，本地统计为 134 局。

两者差别可以这样理解：

- seen：验证任务来自训练分布内的场景，房间/布局更接近训练环境。
- unseen：验证任务来自训练分布外的场景，更考验泛化能力。

后续任务三明确要求跑 `eval_out_of_distribution` 全量，所以真正要报告的主要结果是 unseen 134 局成功率。

## 12. 六类任务为什么难度不同

ALFWorld 的六类任务不是同等难度。

最简单的是 `pick_and_place_simple`：找到物体，拿起来，放到目标位置。

`clean`、`heat`、`cool` 类更难，因为它们在取放之外多了一步状态改变：

```text
拿物体 -> 找工具/装置 -> clean/heat/cool -> 找目标容器 -> 放置
```

其中 heat/cool 常更难，因为 microwave/fridge 可能需要开关，位置也不一定显眼；模型容易犯“拿到了就直接放”或者“去了微波炉但没加热”的规划错误。

`look_at_obj_in_light` 的难点是要让目标物体处于光源条件下，不只是把它放到某处。

`pick_two_obj_and_place` 的难点是记忆和重复控制。模型需要找两个同类物体，容易找到一个后忘记第二个，或者反复检查同一个位置。

## 13. `admissible_commands` 到底算不算作弊

`info["admissible_commands"]` 是当前状态下环境给出的合法动作列表。比如当前 reset 后可能给出：

```text
go to bed 1
go to desk 1
go to drawer 1
...
```

它对人工试玩很友好，对调试也很重要。但如果把完整列表塞进 LLM prompt，模型就不需要自己生成开放式动作，只需要从候选动作里挑一个，难度明显降低。因此这属于“候选动作评测设定”，不能和裸 ReAct 直接比较。

后续 Task 3 建议这样处理：

- prompt 不提供完整 `admissible_commands`，保持与 ReAct baseline 更接近。
- 评测程序内部使用它判断模型动作是否合法，统计非法动作率。
- 如果后面做消融，可以单独开一个 “with admissible commands” 对照实验，并在报告中明确说明它是更强设定。

老师如果追问“那你 `env_check.py` 为什么用了 admissible commands”，可以这样答：

> Task 2 的 `env_check.py` 不是评测智能体能力，只是验证环境 step 接口。因此用 admissible commands 选一个合法动作，是为了让 smoke test 稳定。Task 3 评测模型时不会把完整合法动作列表默认塞进 prompt。

## 14. `Your task is to` 是怎么来的

观察里的：

```text
Your task is to: put a mug in desk.
```

不是模型生成的，也不是 `env_check.py` 拼出来的。它已经写在对应 `game.tw-pddl` 的 grammar/task 字段里。这个 gamefile 又是由 ALFWorld 根据 ALFRED 的 `traj_data.json`、PDDL 初始状态和任务模板生成的。

所以同一个 `game.tw-pddl` 每次 reset，目标任务是一样的；如果环境 reset 到不同 gamefile，任务才会变。

这个问题可以这样汇报：

> ALFWorld 的文本任务目标来自 gamefile 内部的 grammar/task 定义。reset 同一个 gamefile 时任务固定；评测多个 gamefile 时，每个 gamefile 对应不同任务实例。

## 15. 手动试玩怎么做

执行：

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=~/.cache/alfworld
alfworld-play-tw
```

这里使用文档默认的数据目录 `~/.cache/alfworld`；当前服务器上该目录已指向已下载的数据。进入游戏后，先读目标句子 `Your task is to: ...`，再根据房间观察输入动作。

这局目标大概是：把 mug 放到 desk。基本思路不是背答案，而是按 ALFWorld 的通用流程走：

```text
先观察房间有哪些容器和地点
去可能有 mug 的地方，例如 desk、drawer、shelf 等
必要时 open 容器
找到 mug 后 take
去目标 desk
put mug 1 in/on desk ?
```

常见动作长这样：

```text
go to fridge 1
open fridge 1
take apple 1 from fridge 1
go to microwave 1
heat apple 1 with microwave 1
go to countertop 1
put apple 1 in/on countertop 1
```

如果是取放任务，不需要 heat/cool/clean；如果目标写着 hot/cool/clean，就一定要先完成对应处理。命令行交互支持自动补全，遇到不确定的物体编号时可以多看当前观察和合法动作提示。玩到出现 `You won!` 后，输入：

```bash
exit
```

如果需要重新留证，可以临时用 `script` 录制终端过程；但原始终端记录通常会包含 ANSI 控制符、自动补全字符和服务器登录提示，直接提交可读性很差。因此最终交付只保留整理后的 `logs/manual_play_task2_clean.log`，其中包含完整任务目标、动作过程和成功标记 `You won!`。

如果 play 时显示没有 data，优先检查当前是否在 `alfworld` conda 环境，以及 `~/.cache/alfworld` 是否存在 `json_2.1.1` 和 `logic` 目录。当前服务器上默认路径已经可用。


## 16. 老师可能问的问题与回答口径

### Q1：任务二为什么要手动玩一局？

为了建立对 benchmark 的直觉。只看代码不知道环境反馈长什么样，也不知道动作必须多具体。手动玩一局可以理解任务目标、房间观察、合法动作、错误动作反馈和完成条件。

### Q2：`env_check.py` 和手动试玩是不是重复？

不重复。手动试玩证明人能理解环境；`env_check.py` 证明 Python 程序能控制环境。后续 LLM Agent 依赖的是 Python API，所以两者都需要。

### Q3：为什么 `env_check.py` 没有玩通？

因为任务二的最小代码验证只要求 reset 和 step，目标是检查环境接口。玩通一局属于人工交互 log；自动玩通和统计成功率属于任务三。

### Q4：为什么要保存 `env_check_task2.log`？

报告里说“跑通了”不如保留真实运行输出。log 能证明当时的版本、数据路径、split 数量、reset 观察和 step 结果。

### Q5：为什么 unseen 是 134 局？

因为按 ALFWorld 文本环境的过滤规则，`valid_unseen` 中可玩且可解的 `game.tw-pddl` 数量是 134。任务三要求跑 `eval_out_of_distribution` 全量，也就是这 134 局。

### Q6：为什么不用视觉环境？

本练习关注文本版 ALFWorld 和 LLM ReAct 循环。视觉 THOR 环境会引入额外依赖和图像输入，不是本次任务目标。

### Q7：如果把 admissible commands 给模型，会发生什么？

模型动作合法率会变高，但评测难度也下降。它变成“从候选动作里选动作”的设定，不再是裸自然语言动作生成。可以作为消融实验，但主实验需要明确是否使用。

## 17. 当前状态

已经完成：

- `alfworld` conda 环境创建。
- `alfworld==0.4.2`、`textworld==1.7.0`、`PyYAML==6.0.3` 安装。
- ALFWorld 数据下载到 `/root/.cache/alfworld`。
- `env_check.py` 成功跑通 unseen reset/step。
- 程序化验证日志保存到 `logs/env_check_task2.log`。
- 学习版和提交版报告已创建。

手动试玩已完成：

- `logs/manual_play_task2_clean.log` 可读交互记录已生成。
- log 中包含任务目标 `Your task is to: examine the pillow with the desklamp.`。
- log 中包含成功标记 `You won!`。
- 提交版报告中的“手动交互记录”小节已更新，只引用最终 clean log。

## 18. 参考资料

- ALFWorld 官方仓库与 quickstart: https://github.com/alfworld/alfworld
- ALFWorld 官方 `base_config.yaml`: https://github.com/alfworld/alfworld/blob/master/configs/base_config.yaml
- ALFWorld agents README/config 说明: https://github.com/alfworld/alfworld/blob/master/alfworld/agents/README.md
- ReAct ALFWorld notebook: https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb
- ReAct ALFWorld prompts: https://github.com/ysymyth/ReAct/blob/master/prompts/alfworld_3prompts.json
