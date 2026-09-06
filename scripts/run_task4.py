#!/usr/bin/env python3
"""Run the preregistered Task 4 ALFWorld ablation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import TASK_ORDER, gamefile_set_sha256, get_task_type_from_gamefile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task4_experiments.yaml")
    parser.add_argument(
        "--stage",
        choices=["dry-run", "smoke", "primary", "replicate", "all"],
        default="dry-run",
    )
    parser.add_argument("--output-root", default="results/task4_prompt_ablation")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--experiments", nargs="*", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config.get("version") != 1:
        raise ValueError("Unsupported task4 experiment config version")
    experiments = config.get("experiments", {})
    if "baseline" not in experiments or len(experiments) != 11:
        raise ValueError("Task 4 config must contain baseline plus ten variants")
    defaults = config["defaults"]
    baseline = experiments["baseline"].get("overrides", {})
    if baseline:
        raise ValueError("Baseline must not override common defaults")
    for name, item in experiments.items():
        overrides = item.get("overrides", {})
        if name != "baseline" and len(overrides) != 1:
            raise ValueError(f"{name} must change exactly one factor")
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise KeyError(f"{name} has unknown overrides: {sorted(unknown)}")
    fixed_keys = experiments["random_nonmatching_2shot"].get("fixed_prompt_keys", {})
    if set(fixed_keys) != set(TASK_ORDER):
        raise ValueError("Random nonmatching two-shot must freeze prompt keys for all task types")
    if any(len(keys) != 2 for keys in fixed_keys.values()):
        raise ValueError("Each random nonmatching task type must have exactly two prompt keys")
    return config


def check_service(base_url: str) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    with opener.open(request, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"Model service returned HTTP {response.status}")


def cli_args(settings: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in settings.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            result.append(flag)
        else:
            result.extend([flag, str(value)])
    return result


def run_is_complete(output_dir: Path) -> bool:
    required = ["trajectories.jsonl", "summary.json", "metrics_by_task.csv", "run_metadata.json"]
    return all((output_dir / name).is_file() for name in required)


def execute_run(
    config: dict[str, Any],
    name: str,
    seed: int,
    output_root: Path,
    *,
    workers: int | None,
    manifest_override: Path | None = None,
    dry_run: bool = False,
) -> None:
    experiment = config["experiments"][name]
    settings = dict(config["defaults"])
    settings.update(experiment.get("overrides", {}))
    settings["seed"] = seed
    if workers is not None:
        settings["workers"] = workers
    if manifest_override is not None:
        settings["gamefile_manifest"] = str(manifest_override)

    output_dir = output_root / "runs" / name / f"seed_{seed}"
    settings["output_dir"] = str(output_dir)
    command = [sys.executable, str(ROOT / "eval.py"), *cli_args(settings)]
    if dry_run:
        print(" ".join(command))
        return
    if run_is_complete(output_dir):
        print(f"skip complete run: {output_dir}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Incomplete output exists at {output_dir}; inspect it before rerunning"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()

    metadata = {
        "experiment": name,
        "group": experiment["group"],
        "hypothesis": experiment["hypothesis"],
        "seed": seed,
        "settings": settings,
        "command": command,
        "started_at_utc": started_at,
        "elapsed_sec": time.perf_counter() - started,
        "return_code": return_code,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if return_code:
        raise RuntimeError(f"{name} seed {seed} failed with exit code {return_code}")
    if not run_is_complete(output_dir):
        raise RuntimeError(f"{name} seed {seed} did not produce all required files")


def rank_primary(config: dict[str, Any], output_root: Path) -> tuple[str, str]:
    seed = int(config["primary_seed"])
    ranked = []
    for name in config["experiments"]:
        summary_path = output_root / "runs" / name / f"seed_{seed}" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing primary result: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        item = summary["overall"]
        key = (
            -float(item["success_rate"]),
            float(item["invalid_action_rate"]),
            float(item["parse_failure_rate"]),
            float(item["avg_total_tokens"]),
            name,
        )
        ranked.append((key, name))
    ranked.sort()
    return ranked[0][1], ranked[-1][1]


def smoke_manifest(config: dict[str, Any], temp_dir: Path) -> Path:
    manifest_path = ROOT / config["defaults"]["gamefile_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(manifest["source_trajectories"])
    if not source.is_absolute():
        source = manifest_path.parent / source
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = []
    for task in TASK_ORDER:
        candidates = [row for row in rows if row["task_type"] == task]
        selected.append(max(candidates, key=lambda row: row["agent_turns"])["gamefile"])
    game_files = sorted(selected)
    smoke_path = temp_dir / "task4_smoke_manifest.json"
    smoke_path.write_text(
        json.dumps(
            {
                "version": 1,
                "split": "eval_out_of_distribution",
                "gamefiles": game_files,
                "expected_count": len(game_files),
                "gamefile_set_sha256": gamefile_set_sha256(game_files),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if {get_task_type_from_gamefile(path) for path in game_files} != set(TASK_ORDER):
        raise RuntimeError("Smoke manifest does not cover all six task types")
    return smoke_path


def selected_experiments(config: dict[str, Any], requested: list[str] | None) -> list[str]:
    names = list(config["experiments"])
    if requested is None:
        return names
    unknown = set(requested) - set(names)
    if unknown:
        raise KeyError(f"Unknown experiments: {sorted(unknown)}")
    return [name for name in names if name in requested]


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    config = load_config(config_path)
    output_root = ROOT / args.output_root
    names = selected_experiments(config, args.experiments)

    if args.stage == "dry-run":
        for name in names:
            execute_run(
                config,
                name,
                int(config["primary_seed"]),
                output_root,
                workers=args.workers,
                dry_run=True,
            )
        return

    check_service(config["defaults"]["base_url"])

    if args.stage == "smoke":
        smoke_root = ROOT / "results/task4_smoke"
        with tempfile.TemporaryDirectory(prefix="task4-smoke-") as temp:
            manifest = smoke_manifest(config, Path(temp))
            for name in names:
                execute_run(
                    config,
                    name,
                    int(config["primary_seed"]),
                    smoke_root,
                    workers=1,
                    manifest_override=manifest,
                )
        return

    if args.stage in {"primary", "all"}:
        for name in names:
            execute_run(
                config,
                name,
                int(config["primary_seed"]),
                output_root,
                workers=args.workers,
            )

    if args.stage in {"replicate", "all"}:
        best, worst = rank_primary(config, output_root)
        if args.stage == "replicate" and args.experiments is not None:
            replicate_names = names
            print("replicating requested experiments=" + ",".join(replicate_names))
        else:
            replicate_names = [best, worst]
            print(f"replicating best={best} worst={worst}")
        for name in replicate_names:
            for seed in config["replication_seeds"]:
                execute_run(
                    config,
                    name,
                    int(seed),
                    output_root,
                    workers=args.workers,
                )

if __name__ == "__main__":
    main()
