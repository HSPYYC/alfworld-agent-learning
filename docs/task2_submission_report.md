# 任务二：部署 ALFWorld 文本环境

## 1. 目标与环境

本任务部署 ALFWorld 的 TextWorld 文本环境，用于后续在 unseen split 上运行 ReAct 智能体。当前只安装 text-only 版本，不安装 THOR 视觉环境。

| 项目 | 配置 |
| --- | --- |
| Conda 环境 | `alfworld` |
| Python | 3.10.21 |
| ALFWorld | 0.4.2 |
| TextWorld | 1.7.0 |
| 数据目录 | `/root/.cache/alfworld` |
| 配置文件 | `configs/base_config.yaml` |
| 验证脚本 | `env_check.py` |

## 2. 部署方式

```bash
cd /root/alfworld-agent
bash scripts/setup_task2_env.sh
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
```

如需从零下载数据：

```bash
DOWNLOAD_DATA=1 bash scripts/setup_task2_env.sh
```

本机由于代理访问国内 Conda/PyPI 镜像不稳定，Conda 创建环境时使用 `--override-channels -c defaults`，pip 安装使用 `https://pypi.org/simple`。

## 3. 程序化验证结果

验证命令：

```bash
cd /root/alfworld-agent
conda activate alfworld
export ALFWORLD_DATA=/root/.cache/alfworld
python env_check.py --config configs/base_config.yaml --split eval_out_of_distribution
```

完整输出保存于 `logs/env_check_task2.log`。验证脚本完成了三件事：读取配置、统计 train/seen/unseen split 的可玩且可解游戏数量、初始化 unseen 环境并执行一步合法动作。

| Split | 可玩且可解游戏数 | 说明 |
| --- | ---: | --- |
| train | 3553 | 训练 split |
| eval_in_distribution / valid_seen | 140 | seen 验证集 |
| eval_out_of_distribution / valid_unseen | 134 | unseen 验证集 |

unseen split 的任务类型分布：

| 任务类型 | 数量 |
| --- | ---: |
| look_at_obj_in_light | 18 |
| pick_and_place_simple | 24 |
| pick_clean_then_place_in_recep | 31 |
| pick_cool_then_place_in_recep | 21 |
| pick_heat_then_place_in_recep | 23 |
| pick_two_obj_and_place | 17 |

一次 smoke test reset 到的任务为：`put a mug in desk`。环境返回了合法动作列表，脚本执行 `go to bed 1` 后得到新观测，`reward=0`、`done=False`，说明 reset/step 链路正常。

## 4. 手动交互记录

已按要求手动玩通一局。完整可读交互记录保存在 `logs/manual_play_task2_clean.log`。该记录保留了任务目标、关键动作序列和成功标记，并去除了终端控制符等与实验结果无关的内容。

| 项目 | 结果 |
| --- | --- |
| 完整交互记录 | `logs/manual_play_task2_clean.log` |
| 任务目标 | `Your task is to: examine the pillow with the desklamp.` |
| 完成标记 | `You won!` |

本次人工交互体现了 ALFWorld 的基本流程：先阅读 `Your task is to: ...`，再根据房间文本观察探索地点和物体，输入符合 TextWorld grammar 的动作，最后通过与目标物体/装置交互完成任务。

## 5. 需要回答的问题

### eval_in_distribution 和 eval_out_of_distribution

`eval_in_distribution` 对应 `valid_seen`，本地统计为 140 局；`eval_out_of_distribution` 对应 `valid_unseen`，本地统计为 134 局。seen/unseen 的核心差别是测试场景是否与训练场景同分布：seen 更偏向已见房间/布局中的新任务实例，unseen 更强调对新房间和新物体摆放的泛化。后续 ReAct 评测通常报告 unseen split，本练习任务三也明确要求跑 `eval_out_of_distribution` 全量。

### 六类任务难度差别

六类任务从简单取放到带中间处理步骤逐渐变难：`pick_and_place` 只需找物体并放到目标容器；`clean/heat/cool_then_place` 需要先找到物体，再找到对应工具或装置，完成状态改变后再放置；`look_at_obj_in_light` 需要拿物体并与光源交互；`pick_two_obj_and_place` 需要连续找两个同类物体，记忆负担更重。heat/cool 通常更难，因为它们需要额外定位 microwave/fridge 等装置，并正确保持“先处理、后放置”的顺序。

### admissible_commands 是否放进 prompt

`info["admissible_commands"]` 是 TextWorld 给出的当前合法动作集合。把完整合法动作列表直接放进 LLM prompt 会显著降低动作生成难度，因此属于一种更强的候选动作设定；如果实验目标是复现裸 ReAct，应避免在 prompt 中提供完整列表，只在评估脚本内部用它判断非法动作、统计错误，或用于人工试玩的自动补全。后续任务三默认采用不把完整 admissible commands 放进 prompt 的设置，并在结果里统计非法动作率。

### `Your task is to` 的来源

`Your task is to: ...` 来自每个 `game.tw-pddl` 内部的 grammar/task 字段，由 ALFWorld 根据对应 `traj_data.json` 和 PDDL 问题生成。对同一个 gamefile，任务目标是固定的；重复 reset 不会换成另一个任务。评测 split 中切换任务是因为环境在多个 gamefile 之间 reset。

## 6. 参考资料

- ALFWorld 官方仓库与 quickstart: https://github.com/alfworld/alfworld
- ALFWorld 官方 `base_config.yaml`: https://github.com/alfworld/alfworld/blob/master/configs/base_config.yaml
- ReAct ALFWorld notebook: https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb
- ReAct ALFWorld prompts: https://github.com/ysymyth/ReAct/blob/master/prompts/alfworld_3prompts.json
