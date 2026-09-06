#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import yaml


TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}


def ensure_data_dir(data_dir: str | None) -> str:
    if data_dir:
        resolved = data_dir
    elif os.environ.get("ALFWORLD_DATA"):
        resolved = os.environ["ALFWORLD_DATA"]
    elif Path("/root/.cache/alfworld").exists():
        resolved = "/root/.cache/alfworld"
    else:
        resolved = str(Path.home() / ".cache" / "alfworld")

    os.environ["ALFWORLD_DATA"] = resolved
    return resolved


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_env_class(env_type: str):
    import alfworld.agents.environment as environment

    env_class = getattr(environment, env_type, None)
    if env_class is None and hasattr(environment, "get_environment"):
        env_class = environment.get_environment(env_type)
    if env_class is None:
        raise ValueError(f"Unknown ALFWorld environment type: {env_type}")
    return env_class


def selected_task_types(config: dict) -> set[str]:
    return {TASK_TYPES[i] for i in config["env"]["task_types"] if i in TASK_TYPES}


def split_paths(config: dict) -> dict[str, Path]:
    dataset = config["dataset"]
    return {
        "train": Path(os.path.expandvars(dataset["data_path"])),
        "eval_in_distribution": Path(os.path.expandvars(dataset["eval_id_data_path"])),
        "eval_out_of_distribution": Path(os.path.expandvars(dataset["eval_ood_data_path"])),
    }


def count_games(data_path: Path, allowed_task_types: set[str]) -> tuple[int, Counter, Counter]:
    task_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    total = 0

    for traj_file in sorted(data_path.rglob("traj_data.json")):
        root = traj_file.parent
        root_text = str(root)

        if "movable" in root_text or "Sliced" in root_text:
            skipped["movable_or_sliced"] += 1
            continue

        with traj_file.open("r", encoding="utf-8") as f:
            traj_data = json.load(f)

        task_type = traj_data.get("task_type")
        if task_type not in allowed_task_types:
            skipped["task_type_filtered"] += 1
            continue

        game_file = root / "game.tw-pddl"
        if not game_file.exists():
            skipped["missing_game"] += 1
            continue

        with game_file.open("r", encoding="utf-8") as f:
            game_data = json.load(f)

        if "solvable" not in game_data:
            skipped["missing_solvable"] += 1
            continue
        if not game_data["solvable"]:
            skipped["unsolvable"] += 1
            continue

        total += 1
        task_counts[task_type] += 1

    return total, task_counts, skipped


def print_dataset_summary(config: dict) -> None:
    allowed = selected_task_types(config)
    print("Dataset summary:")
    for split, path in split_paths(config).items():
        total, task_counts, skipped = count_games(path, allowed)
        print(f"- {split}: {total} playable solvable games")
        for task_name, count in sorted(task_counts.items()):
            print(f"  - {task_name}: {count}")
        if skipped:
            skipped_text = ", ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
            print(f"  - skipped: {skipped_text}")


def pick_safe_action(commands: list[str]) -> str:
    for command in commands:
        if command.startswith("go to "):
            return command
    return commands[0]


def as_scalar(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    try:
        return value[0]
    except Exception:
        return value


def run_env_check(config: dict, split: str, batch_size: int) -> None:
    env_class = get_env_class(config["env"]["type"])
    env = env_class(config, train_eval=split)
    env = env.init_env(batch_size=batch_size)

    obs, info = env.reset()
    commands = info["admissible_commands"][0]
    gamefile = info.get("extra.gamefile", [None])[0]

    print()
    print("Environment reset:")
    print(f"- split: {split}")
    print(f"- gamefile: {gamefile}")
    print("- observation:")
    print(obs[0].strip())
    print()
    print("First 10 admissible commands:")
    for command in commands[:10]:
        print(f"- {command}")

    action = pick_safe_action(commands)
    obs, reward, done, info = env.step([action])

    print()
    print("One-step smoke test:")
    print(f"- action: {action}")
    print(f"- reward: {as_scalar(reward)}")
    print(f"- done: {as_scalar(done)}")
    print("- next observation:")
    print(obs[0].strip())

    close = getattr(env, "close", None)
    if close is not None:
        close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the ALFWorld text environment.")
    parser.add_argument("--config", default="configs/base_config.yaml", help="Path to the ALFWorld config YAML.")
    parser.add_argument(
        "--split",
        default="eval_out_of_distribution",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        help="Dataset split to initialize.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the TextWorld environment.")
    parser.add_argument("--data-dir", default=None, help="Override ALFWORLD_DATA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = ensure_data_dir(args.data_dir)
    config = load_config(Path(args.config))

    print("Python ALFWorld check")
    print(f"- alfworld: {version('alfworld')}")
    print(f"- textworld: {version('textworld')}")
    print(f"- ALFWORLD_DATA: {data_dir}")
    print(f"- config: {args.config}")
    print()

    print_dataset_summary(config)
    run_env_check(config, args.split, args.batch_size)


if __name__ == "__main__":
    main()
