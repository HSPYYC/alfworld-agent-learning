#!/usr/bin/env python3
"""Aggregate the look_at_obj_in_light lamp-repair diagnostic runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = [
    "baseline_look_only",
    "repair_look_only",
    "grammar_hint_look_only",
    "grammar_hint_plus_repair_look_only",
]
TABLE_COLUMNS = [
    "experiment",
    "successes",
    "episodes",
    "success_rate",
    "avg_env_steps",
    "invalid_action_rate",
    "parse_failure_rate",
    "lookat_lamp_repairs",
    "avg_total_tokens",
    "wall_time_sec",
]
COMPARISONS = [
    (
        "baseline-look-only vs repair-look-only",
        "baseline_look_only",
        "repair_look_only",
        "parser/action-repair 是否有效",
    ),
    (
        "baseline-look-only vs grammar-hint-look-only",
        "baseline_look_only",
        "grammar_hint_look_only",
        "prompt-only 是否有效",
    ),
    (
        "grammar-hint-look-only vs grammar-hint+repair",
        "grammar_hint_look_only",
        "grammar_hint_plus_repair_look_only",
        "组合上限，不做单变量归因",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/task4_lookat_lamp_repair")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    return parser.parse_args()


def read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value: Any) -> str:
    return f"{float(value):.6f}"


def fmt_rate(value: Any) -> str:
    return f"{100 * float(value):.1f}%"


def experiment_row(name: str, summary: dict[str, Any]) -> dict[str, str]:
    overall = summary["overall"]
    return {
        "experiment": name,
        "successes": str(overall["successes"]),
        "episodes": str(overall["episodes"]),
        "success_rate": fmt_float(overall["success_rate"]),
        "avg_env_steps": fmt_float(overall["avg_env_steps"]),
        "invalid_action_rate": fmt_float(overall["invalid_action_rate"]),
        "parse_failure_rate": fmt_float(overall["parse_failure_rate"]),
        "lookat_lamp_repairs": str(overall.get("lookat_lamp_repairs", 0)),
        "avg_total_tokens": fmt_float(overall["avg_total_tokens"]),
        "wall_time_sec": fmt_float(summary.get("wall_time_sec", 0.0)),
    }


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], columns: list[str], right_align: set[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    align = ["---:" if column in right_align else "---" for column in columns]
    sep = "| " + " | ".join(align) + " |"
    body = ["| " + " | ".join(row.get(column, "") for column in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def comparison_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for label, before_name, after_name, conclusion_hint in COMPARISONS:
        if before_name not in summaries or after_name not in summaries:
            rows.append(
                {
                    "对比": label,
                    "Before": "missing",
                    "After": "missing",
                    "Delta Success Rate": "",
                    "Delta Invalid Rate": "",
                    "结论": conclusion_hint,
                }
            )
            continue
        before = summaries[before_name]["overall"]
        after = summaries[after_name]["overall"]
        delta_success = float(after["success_rate"]) - float(before["success_rate"])
        delta_invalid = float(after["invalid_action_rate"]) - float(before["invalid_action_rate"])
        rows.append(
            {
                "对比": label,
                "Before": f"{before['successes']}/{before['episodes']} ({fmt_rate(before['success_rate'])})",
                "After": f"{after['successes']}/{after['episodes']} ({fmt_rate(after['success_rate'])})",
                "Delta Success Rate": f"{100 * delta_success:+.1f} pp",
                "Delta Invalid Rate": f"{100 * delta_invalid:+.1f} pp",
                "结论": conclusion_hint,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    results_root = ROOT / args.results_root
    analysis_dir = results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    rows = []
    for name in args.experiments:
        run_dir = results_root / name
        if not (run_dir / "summary.json").is_file():
            continue
        summary = read_summary(run_dir)
        summaries[name] = summary
        rows.append(experiment_row(name, summary))

    csv_path = analysis_dir / "lookat_before_after.csv"
    write_csv(csv_path, rows, TABLE_COLUMNS)

    comp = comparison_rows(summaries)
    md_path = analysis_dir / "lookat_before_after.md"
    md = [
        "# look_at_obj_in_light 专项补跑",
        "",
        markdown_table(rows, TABLE_COLUMNS, set(TABLE_COLUMNS) - {"experiment"}),
        "",
        "## 前后对比",
        "",
        markdown_table(
            comp,
            ["对比", "Before", "After", "Delta Success Rate", "Delta Invalid Rate", "结论"],
            {"Before", "After", "Delta Success Rate", "Delta Invalid Rate"},
        ),
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
