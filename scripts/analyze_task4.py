#!/usr/bin/env python3
"""Validate, aggregate, and plot Task 4 ablation results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK_ORDER = [
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
]
LABEL_MAP = {
    "动作格式/任务语义错误": "action_format_or_task_protocol",
    "规划/前置条件错误": "planning_or_precondition",
    "探索/找物错误": "exploration_or_search",
    "感知/记忆错误": "perception_or_memory",
    "死循环/错误恢复失败": "loop_or_recovery",
}
RANK_METRICS = [
    "success_rate",
    "invalid_action_rate",
    "parse_failure_rate",
    "avg_total_tokens",
]
AGGREGATE_METRICS = [
    "success_rate",
    "avg_env_steps",
    "avg_success_env_steps",
    "avg_agent_turns",
    "avg_llm_calls",
    "invalid_action_rate",
    "parse_failure_rate",
    "repeated_action_rate",
    "avg_prompt_tokens",
    "avg_completion_tokens",
    "avg_total_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task4_experiments.yaml")
    parser.add_argument("--results-root", default="results/task4_prompt_ablation")
    parser.add_argument(
        "--failure-analysis",
        default="results/task3_react_baseline/failure_analysis.csv",
    )
    parser.add_argument(
        "--baseline-trajectories",
        default="results/task3_react_baseline/trajectories.jsonl",
    )
    parser.add_argument(
        "--failure-annotations",
        default="configs/task4_failure_annotations.csv",
    )
    parser.add_argument("--failure-seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gamefile_hash(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(sorted(row["gamefile"] for row in rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


def validate_run(
    run_dir: Path,
    expected_count: int,
    expected_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(run_dir / "trajectories.jsonl")
    errors: list[str] = []
    if len(rows) != expected_count:
        errors.append(f"expected {expected_count} rows, found {len(rows)}")
    if len({row["gamefile"] for row in rows}) != len(rows):
        errors.append("duplicate gamefiles")
    actual_hash = gamefile_hash(rows)
    if actual_hash != expected_hash:
        errors.append(f"gamefile hash mismatch: {actual_hash}")
    recomputed = {
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "env_steps": sum(int(row["env_steps"]) for row in rows),
        "agent_turns": sum(int(row["agent_turns"]) for row in rows),
        "action_steps": sum(int(row["action_steps"]) for row in rows),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in rows),
        "parse_failures": sum(int(row["parse_failures"]) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in rows),
    }
    overall = summary["overall"]
    for key, value in recomputed.items():
        if int(overall[key]) != value:
            errors.append(f"{key}: summary={overall[key]} recomputed={value}")
    expected_success_rate = recomputed["successes"] / len(rows) if rows else 0.0
    expected_invalid_rate = (
        recomputed["invalid_actions"] / recomputed["action_steps"]
        if recomputed["action_steps"]
        else 0.0
    )
    expected_parse_rate = (
        recomputed["parse_failures"] / recomputed["agent_turns"]
        if recomputed["agent_turns"]
        else 0.0
    )
    for key, expected in [
        ("success_rate", expected_success_rate),
        ("invalid_action_rate", expected_invalid_rate),
        ("parse_failure_rate", expected_parse_rate),
    ]:
        if not close(float(overall[key]), expected):
            errors.append(f"{key}: summary={overall[key]} recomputed={expected}")
    return summary, rows, errors


def ranking_key(name: str, summary: dict[str, Any]) -> tuple[Any, ...]:
    item = summary["overall"]
    return (
        -float(item["success_rate"]),
        float(item["invalid_action_rate"]),
        float(item["parse_failure_rate"]),
        float(item["avg_total_tokens"]),
        name,
    )


def load_formal_runs(
    config: dict[str, Any],
    results_root: Path,
) -> tuple[dict[tuple[str, int], tuple[dict[str, Any], list[dict[str, Any]]]], list[dict[str, Any]]]:
    manifest_path = ROOT / config["defaults"]["gamefile_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = int(manifest["expected_count"])
    expected_hash = str(manifest["gamefile_set_sha256"])
    loaded = {}
    checks = []
    seeds = [int(config["primary_seed"]), *map(int, config["replication_seeds"])]
    for name in config["experiments"]:
        for seed in seeds:
            run_dir = results_root / "runs" / name / f"seed_{seed}"
            if not (run_dir / "summary.json").is_file():
                continue
            summary, rows, errors = validate_run(run_dir, expected_count, expected_hash)
            loaded[(name, seed)] = (summary, rows)
            checks.append(
                {
                    "experiment": name,
                    "seed": seed,
                    "episodes": len(rows),
                    "gamefile_set_sha256": gamefile_hash(rows),
                    "errors": len(errors),
                    "status": "pass" if not errors else "fail",
                    "details": "; ".join(errors),
                }
            )
    return loaded, checks


def primary_tables(
    config: dict[str, Any],
    loaded: dict[tuple[str, int], tuple[dict[str, Any], list[dict[str, Any]]]],
    analysis_dir: Path,
) -> tuple[str, str]:
    seed = int(config["primary_seed"])
    baseline_summary, baseline_rows = loaded[("baseline", seed)]
    baseline_success = float(baseline_summary["overall"]["success_rate"])
    table = []
    by_task_table = []
    paired = []
    baseline_by_game = {row["game_id"]: row for row in baseline_rows}

    for name, experiment in config["experiments"].items():
        summary, rows = loaded[(name, seed)]
        item = summary["overall"]
        table.append(
            {
                "experiment": name,
                "display_name_zh": experiment.get("display_name_zh", name),
                "direction_zh": experiment.get("direction_zh", experiment["group"]),
                "group": experiment["group"],
                "hypothesis": experiment["hypothesis"],
                "seed": seed,
                "successes": item["successes"],
                "episodes": item["episodes"],
                "success_rate": item["success_rate"],
                "delta_success_rate_pp": 100 * (float(item["success_rate"]) - baseline_success),
                "avg_env_steps": item["avg_env_steps"],
                "avg_success_env_steps": item["avg_success_env_steps"],
                "avg_agent_turns": item["avg_agent_turns"],
                "avg_llm_calls": item["avg_llm_calls"],
                "invalid_action_rate": item["invalid_action_rate"],
                "parse_failure_rate": item["parse_failure_rate"],
                "repeated_action_rate": item["repeated_action_rate"],
                "avg_prompt_tokens": item["avg_prompt_tokens"],
                "avg_completion_tokens": item["avg_completion_tokens"],
                "avg_total_tokens": item["avg_total_tokens"],
                "errors": item["errors"],
            }
        )
        for task in TASK_ORDER:
            task_item = summary["by_task"][task]
            by_task_table.append(
                {
                    "experiment": name,
                    "display_name_zh": experiment.get("display_name_zh", name),
                    "direction_zh": experiment.get("direction_zh", experiment["group"]),
                    "seed": seed,
                    "task_type": task,
                    **{key: task_item[key] for key in AGGREGATE_METRICS},
                    "successes": task_item["successes"],
                    "episodes": task_item["episodes"],
                    "errors": task_item["errors"],
                }
            )

        rows_by_game = {row["game_id"]: row for row in rows}
        for task in ["overall", *TASK_ORDER]:
            game_ids = [
                game_id
                for game_id, row in baseline_by_game.items()
                if task == "overall" or row["task_type"] == task
            ]
            transitions = Counter(
                (
                    bool(baseline_by_game[game_id]["success"]),
                    bool(rows_by_game[game_id]["success"]),
                )
                for game_id in game_ids
            )
            paired.append(
                {
                    "experiment": name,
                    "display_name_zh": experiment.get("display_name_zh", name),
                    "direction_zh": experiment.get("direction_zh", experiment["group"]),
                    "task_type": task,
                    "baseline_failure_variant_success": transitions[(False, True)],
                    "baseline_success_variant_failure": transitions[(True, False)],
                    "both_success": transitions[(True, True)],
                    "both_failure": transitions[(False, False)],
                    "net_successes": transitions[(False, True)] - transitions[(True, False)],
                }
            )

    write_csv(analysis_dir / "ablation_results_seed0.csv", table, list(table[0]))
    token_fields = [
        "experiment",
        "display_name_zh",
        "direction_zh",
        "seed",
        "avg_llm_calls",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_total_tokens",
    ]
    write_csv(
        analysis_dir / "token_costs_seed0.csv",
        [{field: row[field] for field in token_fields} for row in table],
        token_fields,
    )
    write_csv(analysis_dir / "metrics_by_task_all_seed0.csv", by_task_table, list(by_task_table[0]))
    write_csv(analysis_dir / "paired_success_changes.csv", paired, list(paired[0]))

    ranked = sorted(
        (
            ranking_key(name, loaded[(name, seed)][0]),
            name,
        )
        for name in config["experiments"]
    )
    best, worst = ranked[0][1], ranked[-1][1]
    (analysis_dir / "best_worst_selection.json").write_text(
        json.dumps(
            {
                "ranking": config["ranking"],
                "ordered_best_to_worst": [name for _, name in ranked],
                "best": best,
                "worst": worst,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best, worst


def multiseed_table(
    config: dict[str, Any],
    loaded: dict[tuple[str, int], tuple[dict[str, Any], list[dict[str, Any]]]],
    best: str,
    worst: str,
    analysis_dir: Path,
) -> None:
    seeds = [int(config["primary_seed"]), *map(int, config["replication_seeds"])]

    def build_rows(names: list[str]) -> list[dict[str, Any]]:
        rows = []
        for name in names:
            summaries = [
                loaded[(name, seed)][0]
                for seed in seeds
                if (name, seed) in loaded
            ]
            if len(summaries) != len(seeds):
                continue
            row: dict[str, Any] = {
                "experiment": name,
                "display_name_zh": config["experiments"][name].get("display_name_zh", name),
                "direction_zh": config["experiments"][name].get("direction_zh", config["experiments"][name]["group"]),
                "group": config["experiments"][name]["group"],
                "hypothesis": config["experiments"][name]["hypothesis"],
                "seeds": ",".join(str(seed) for seed in seeds),
                "runs": len(seeds),
                "successes": ",".join(
                    str(summary["overall"]["successes"]) for summary in summaries
                ),
            }
            for metric in AGGREGATE_METRICS:
                values = [float(summary["overall"][metric]) for summary in summaries]
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values)
            rows.append(row)
        return rows

    all_rows = build_rows(list(config["experiments"]))
    if all_rows:
        write_csv(analysis_dir / "multiseed_all_configs.csv", all_rows, list(all_rows[0]))
    best_worst_rows = [row for row in all_rows if row["experiment"] in {best, worst}]
    if best_worst_rows:
        write_csv(
            analysis_dir / "multiseed_best_worst.csv",
            best_worst_rows,
            list(best_worst_rows[0]),
        )

    by_task_rows = []
    for name in config["experiments"]:
        summaries = [
            loaded[(name, seed)][0]
            for seed in seeds
            if (name, seed) in loaded
        ]
        if len(summaries) != len(seeds):
            continue
        for task in TASK_ORDER:
            row: dict[str, Any] = {
                "experiment": name,
                "display_name_zh": config["experiments"][name].get("display_name_zh", name),
                "direction_zh": config["experiments"][name].get("direction_zh", config["experiments"][name]["group"]),
                "group": config["experiments"][name]["group"],
                "task_type": task,
                "seeds": ",".join(str(seed) for seed in seeds),
                "runs": len(seeds),
                "successes": ",".join(
                    str(summary["by_task"][task]["successes"])
                    for summary in summaries
                ),
            }
            for metric in AGGREGATE_METRICS:
                values = [
                    float(summary["by_task"][task][metric])
                    for summary in summaries
                ]
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values)
            by_task_rows.append(row)
    if by_task_rows:
        write_csv(
            analysis_dir / "multiseed_by_task_all_configs.csv",
            by_task_rows,
            list(by_task_rows[0]),
        )

def persistent_failures(
    config: dict[str, Any],
    loaded: dict[tuple[str, int], tuple[dict[str, Any], list[dict[str, Any]]]],
    analysis_dir: Path,
) -> None:
    seed = int(config["primary_seed"])
    failures = []
    for name in config["experiments"]:
        rows = loaded[(name, seed)][1]
        failures.append({row["game_id"] for row in rows if not row["success"]})
    persistent = set.intersection(*failures)
    baseline_rows = {
        row["game_id"]: row for row in loaded[("baseline", seed)][1]
    }
    output = [
        {
            "game_id": game_id,
            "task_type": baseline_rows[game_id]["task_type"],
            "baseline_termination_reason": baseline_rows[game_id]["termination_reason"],
            "baseline_env_steps": baseline_rows[game_id]["env_steps"],
            "baseline_invalid_actions": baseline_rows[game_id]["invalid_actions"],
            "baseline_parse_failures": baseline_rows[game_id]["parse_failures"],
        }
        for game_id in sorted(persistent)
    ]
    fields = [
        "game_id",
        "task_type",
        "baseline_termination_reason",
        "baseline_env_steps",
        "baseline_invalid_actions",
        "baseline_parse_failures",
    ]
    write_csv(analysis_dir / "persistent_failures_all_seed0_configs.csv", output, fields)


def failure_sample(args: argparse.Namespace, analysis_dir: Path) -> tuple[Counter, Counter]:
    failure_path = ROOT / args.failure_analysis
    with failure_path.open("r", encoding="utf-8") as f:
        failures = sorted(csv.DictReader(f), key=lambda row: row["game_id"])
    if len(failures) != 42:
        raise ValueError(f"Expected 42 baseline failures, found {len(failures)}")
    sample = random.Random(args.failure_seed).sample(failures, 20)

    annotations_path = ROOT / args.failure_annotations
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = {row["failure_id"]: row for row in csv.DictReader(f)}
    missing = [row["failure_id"] for row in sample if row["failure_id"] not in annotations]
    if missing:
        raise ValueError(f"Missing manual annotations for sampled failure IDs: {missing}")

    trajectories = {
        row["game_id"]: row
        for row in read_jsonl(ROOT / args.baseline_trajectories)
    }
    output = []
    primary_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    for sample_index, row in enumerate(sample, start=1):
        trajectory = trajectories.get(row["game_id"])
        if trajectory is None or trajectory["success"]:
            raise ValueError(f"Sample is not a valid failed trajectory: {row['game_id']}")
        annotation = annotations[row["failure_id"]]
        label = LABEL_MAP[row["primary_label"]]
        primary_counts[label] += 1
        termination_counts[row["termination_reason"]] += 1
        output.append(
            {
                "sample_index": sample_index,
                "sample_seed": args.failure_seed,
                "failure_id": row["failure_id"],
                "episode_index": row["episode_index"],
                "game_id": row["game_id"],
                "task_type": row["task_type"],
                "task": row["task"],
                "termination_reason": row["termination_reason"],
                "env_steps": row["env_steps"],
                "agent_turns": row["agent_turns"],
                "invalid_actions": row["invalid_actions"],
                "parse_failures": row["parse_failures"],
                "repeated_actions": row["repeated_actions"],
                "primary_label": label,
                "secondary_label": annotation["secondary_label"],
                "evidence": annotation["evidence"],
                "recommended_fix": annotation["recommended_fix"],
                "confidence": annotation["confidence"],
            }
        )
    write_csv(analysis_dir / "failure_sample_seed42.csv", output, list(output[0]))
    count_rows = [
        {
            "label": label,
            "count": primary_counts[label],
            "fraction": primary_counts[label] / 20,
        }
        for label in LABEL_MAP.values()
    ]
    write_csv(
        analysis_dir / "failure_category_counts.csv",
        count_rows,
        ["label", "count", "fraction"],
    )
    metadata = {
        "algorithm": "random.Random(seed).sample(sorted_by_game_id, 20)",
        "seed": args.failure_seed,
        "population_size": len(failures),
        "sample_size": len(sample),
        "failure_analysis_sha256": hashlib.sha256(failure_path.read_bytes()).hexdigest(),
        "baseline_trajectories_sha256": hashlib.sha256(
            (ROOT / args.baseline_trajectories).read_bytes()
        ).hexdigest(),
        "sample_failure_ids": [int(row["failure_id"]) for row in sample],
    }
    (analysis_dir / "failure_sample_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return primary_counts, termination_counts


def plots(
    config: dict[str, Any],
    loaded: dict[tuple[str, int], tuple[dict[str, Any], list[dict[str, Any]]]],
    primary_counts: Counter,
    termination_counts: Counter,
    analysis_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(LABEL_MAP.values())
    values = [primary_counts[label] for label in labels]
    display = [
        "Action format /\nprotocol",
        "Planning /\nprecondition",
        "Exploration /\nsearch",
        "Perception /\nmemory",
        "Loop /\nrecovery",
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(display, values, color=["#C44E52", "#DD8452", "#55A868", "#4C72B0", "#8172B2"])
    ax.bar_label(bars)
    ax.set_ylabel("Failures in fixed random sample (n=20)")
    ax.set_title("Baseline failure causes")
    ax.set_ylim(0, max(values) + 2)
    fig.tight_layout()
    fig.savefig(analysis_dir / "failure_primary_causes.png", dpi=180)
    plt.close(fig)

    term_labels = sorted(termination_counts)
    term_values = [termination_counts[label] for label in term_labels]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(term_labels, term_values, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_ylabel("Failures in fixed random sample (n=20)")
    ax.set_title("Failure termination mechanisms")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(analysis_dir / "failure_termination_mechanisms.png", dpi=180)
    plt.close(fig)

    seed = int(config["primary_seed"])
    names = list(config["experiments"])
    rates = [100 * loaded[(name, seed)][0]["overall"]["success_rate"] for name in names]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, rates, color="#55A868")
    ax.bar_label(bars, fmt="%.1f")
    ax.set_xlabel("Success rate (%)")
    ax.set_title("Task 4 prompt ablations (seed 0)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(analysis_dir / "ablation_success_rates.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    results_root = ROOT / args.results_root
    analysis_dir = results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    loaded, checks = load_formal_runs(config, results_root)
    primary_seed = int(config["primary_seed"])
    missing_primary = [
        name for name in config["experiments"] if (name, primary_seed) not in loaded
    ]
    if missing_primary:
        raise FileNotFoundError(f"Missing primary results: {missing_primary}")
    failed_checks = [check for check in checks if check["status"] != "pass"]
    write_csv(
        analysis_dir / "consistency_checks.csv",
        checks,
        ["experiment", "seed", "episodes", "gamefile_set_sha256", "errors", "status", "details"],
    )
    if failed_checks:
        raise ValueError(f"Result consistency checks failed: {failed_checks}")

    best, worst = primary_tables(config, loaded, analysis_dir)
    multiseed_table(config, loaded, best, worst, analysis_dir)
    persistent_failures(config, loaded, analysis_dir)
    primary_counts, termination_counts = failure_sample(args, analysis_dir)
    plots(config, loaded, primary_counts, termination_counts, analysis_dir)
    print(f"best={best} worst={worst}")
    print(f"Wrote Task 4 analysis to {analysis_dir}")


if __name__ == "__main__":
    main()
