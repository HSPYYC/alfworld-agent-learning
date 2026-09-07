#!/usr/bin/env python3
"""Run a small MPO-reference-style ReAct probe without using MPO."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval as core  # noqa: E402


REFERENCE_COMMIT = "5529eca70ab352eb7e56533ca44cb60fbf34bef1"
STOP_WORDS = ["\nObservation:", "\nTask:", "\n---"]
SYSTEM_INSTRUCTION = """Interact with a household to solve the task.
At every turn, output exactly one environment action. You may reason first, but the same response must contain exactly one Action line in one of these forms:
Thought: your concise reasoning
Action: your next action

or:
Action: your next action

Available actions:
1. go to {receptacle}
2. take {object} from {receptacle}
3. put {object} in/on {receptacle}
4. open {receptacle}
5. close {receptacle}
6. use {object}
7. clean {object} with {receptacle}
8. heat {object} with {receptacle}
9. cool {object} with {receptacle}

Do not emit Observation yourself. If an action fails, use the next observation to choose a different action."""

REFERENCE_TASK_TYPES = {
    "pick_and_place_simple": "pick_and_place",
    "pick_clean_then_place_in_recep": "pick_clean_then_place",
    "pick_heat_then_place_in_recep": "pick_heat_then_place",
    "pick_cool_then_place_in_recep": "pick_cool_then_place",
    "look_at_obj_in_light": "look_at_obj",
    "pick_two_obj_and_place": "pick_two_obj",
}


@dataclass(frozen=True)
class ProbeArgs:
    strategy: str
    config: str
    prompt_file: str
    manifest: str
    output_dir: str
    base_url: str
    model: str
    api_key: str
    workers: int
    max_steps: int
    max_agent_turns: int
    max_tokens: int
    temperature: float
    timeout: float
    trust_env: bool
    seed: int
    max_consecutive_parse_failures: int


def parse_args() -> ProbeArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["matched", "mixed6"], required=True)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument(
        "--prompt-file", default="prompts/mpo_reference_alfworld_icl.json"
    )
    parser.add_argument(
        "--manifest", default="configs/reference_baseline_probe_manifest.json"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-agent-turns", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--trust-env", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-consecutive-parse-failures", type=int, default=8)
    values = vars(parser.parse_args())
    if values["workers"] < 1:
        parser.error("--workers must be at least 1")
    if values["max_steps"] < 1 or values["max_agent_turns"] < 1:
        parser.error("step and turn limits must be at least 1")
    if values["max_tokens"] < 1:
        parser.error("--max-tokens must be at least 1")
    return ProbeArgs(**values)


def selected_keys(strategy: str, task_type: str) -> tuple[str, ...]:
    if strategy == "matched":
        prefix = core.PROMPT_PREFIXES[task_type]
        return (f"react_{prefix}_0", f"react_{prefix}_1")
    if strategy == "mixed6":
        return tuple(
            f"react_{core.PROMPT_PREFIXES[name]}_0" for name in core.TASK_ORDER
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def reference_example_messages(
    example: list[dict[str, str]], label: str
) -> list[dict[str, str]]:
    if not example or example[0].get("role") != "user":
        raise ValueError("Reference example must begin with a user message")
    messages = [
        {"role": "user", "content": f"{label}\n{example[0].get('content', '').strip()}"}
    ]
    pending_thoughts: list[str] = []
    for index in range(1, len(example), 2):
        if index + 1 >= len(example):
            raise ValueError("Reference example ends without an observation")
        assistant = example[index]
        observation = example[index + 1]
        if assistant.get("role") != "assistant" or observation.get("role") != "user":
            raise ValueError("Reference example must alternate assistant and user messages")
        assistant_content = assistant.get("content", "").strip()
        if not re.search(r"(?:^|\n)\s*Action:", assistant_content, re.IGNORECASE):
            pending_thoughts.append(assistant_content)
            continue
        content: list[str] = []
        if pending_thoughts:
            content.extend(pending_thoughts)
            pending_thoughts.clear()
        content.append(assistant_content)
        messages.append({"role": "assistant", "content": "\n".join(content)})
        messages.append({"role": "user", "content": observation.get("content", "").strip()})
    if pending_thoughts:
        raise ValueError("Example ends with thought but no action")
    return messages


def load_demo_messages(
    prompt_file: Path, strategy: str
) -> tuple[dict[str, list[dict[str, str]]], dict[str, tuple[str, ...]]]:
    raw = json.loads(prompt_file.read_text(encoding="utf-8"))
    messages_by_task = {}
    keys_by_task = {}
    for task_type in core.TASK_ORDER:
        keys = selected_keys(strategy, task_type)
        if strategy == "matched":
            reference_types = [REFERENCE_TASK_TYPES[task_type]] * 2
            example_indices = [0, 1]
        else:
            reference_types = [REFERENCE_TASK_TYPES[name] for name in core.TASK_ORDER]
            example_indices = [0] * len(reference_types)
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        for position, (reference_type, example_index) in enumerate(
            zip(reference_types, example_indices, strict=True), start=1
        ):
            examples = raw.get(reference_type)
            if not isinstance(examples, list) or len(examples) <= example_index:
                raise ValueError(
                    f"Missing reference example {reference_type}[{example_index}]"
                )
            messages.extend(
                reference_example_messages(
                    examples[example_index], f"Example task {position}:"
                )
            )
        messages_by_task[task_type] = messages
        keys_by_task[task_type] = keys
    return messages_by_task, keys_by_task


def parse_reference_action(text: str) -> tuple[str, str | None]:
    matches = re.findall(
        r"(?:^|\n)\s*Action:\s*([^\n]+)", text.strip(), flags=re.IGNORECASE
    )
    if not matches:
        return "", "missing_action_line"
    if len(matches) != 1:
        return "", "multiple_action_lines"
    normalized, kind = core.normalize_action(matches[0])
    if kind != "action":
        return normalized, "invalid_action_content"
    return normalized, None


def load_probe_gamefiles(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = ROOT / manifest["source_manifest"]
    source_files = core.load_gamefile_manifest(source_path)
    source_hash = core.gamefile_set_sha256(source_files)
    if source_hash != manifest["source_gamefile_set_sha256"]:
        raise ValueError(
            f"Source manifest hash mismatch: expected {manifest['source_gamefile_set_sha256']}, "
            f"found {source_hash}"
        )
    grouped: dict[str, list[str]] = defaultdict(list)
    for gamefile in source_files:
        grouped[core.get_task_type_from_gamefile(gamefile)].append(gamefile)
    per_task = int(manifest["per_task"])
    selected = []
    for task_type in core.TASK_ORDER:
        candidates = sorted(grouped[task_type])
        if len(candidates) < per_task:
            raise ValueError(f"Not enough games for {task_type}: {len(candidates)}")
        selected.extend(candidates[:per_task])
    selected = sorted(selected)
    if len(selected) != int(manifest["expected_count"]):
        raise ValueError("Probe manifest expected_count mismatch")
    actual_hash = core.gamefile_set_sha256(selected)
    if actual_hash != manifest["gamefile_set_sha256"]:
        raise ValueError(
            f"Probe gamefile hash mismatch: expected {manifest['gamefile_set_sha256']}, "
            f"found {actual_hash}"
        )
    return selected


def chat_completion(
    client: OpenAI,
    args: ProbeArgs,
    messages: list[dict[str, str]],
    request_seed: int,
) -> core.LLMResult:
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=STOP_WORDS,
        seed=request_seed,
    )
    prompt_tokens, completion_tokens, total_tokens = core.response_usage(response)
    return core.LLMResult(
        text=(response.choices[0].message.content or "").strip(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def run_episode(
    env: Any,
    client: OpenAI,
    args: ProbeArgs,
    demo_messages: dict[str, list[dict[str, str]]],
    keys_by_task: dict[str, tuple[str, ...]],
    index: int,
    expected_gamefile: str,
    allowed_gamefiles: set[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    obs, info = env.reset()
    gamefile = info["extra.gamefile"][0]
    if str(Path(gamefile).resolve()) not in allowed_gamefiles:
        raise RuntimeError(f"Environment loaded a game outside the worker manifest: {gamefile}")
    task_type = core.get_task_type_from_gamefile(gamefile)
    initial_obs = core.normalize_initial_observation(obs[0])
    game_id = core.game_id_from_gamefile(gamefile)
    messages = [dict(message) for message in demo_messages[task_type]]
    messages.append({"role": "user", "content": f"Current task:\n{initial_obs}"})

    trajectory = []
    agent_turns = 0
    env_steps = 0
    action_steps = 0
    thought_steps = 0
    parse_failures = 0
    invalid_actions = 0
    nothing_happens = 0
    repeated_actions = 0
    consecutive_parse_failures = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    last_action = None
    done_flag = False
    won_flag = False
    final_reward = 0
    error_message = None
    termination_reason = "max_agent_turns"
    admissible = core.current_commands(info)

    while env_steps < args.max_steps and agent_turns < args.max_agent_turns:
        agent_turns += 1
        try:
            result = chat_completion(
                client,
                args,
                messages,
                core.stable_request_seed(
                    args.seed, game_id, "reference_chat", agent_turns
                ),
            )
        except Exception as exc:
            error_message = repr(exc)
            termination_reason = "runtime_error"
            break

        raw_output = result.text
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        total_tokens += result.total_tokens
        if re.search(r"(?:^|\n)\s*Thought:", raw_output, flags=re.IGNORECASE):
            thought_steps += 1
        normalized, parse_failure_reason = parse_reference_action(raw_output)
        messages.append({"role": "assistant", "content": raw_output})

        in_admissible = None
        invalid_action = False
        if parse_failure_reason:
            parse_failures += 1
            consecutive_parse_failures += 1
            observation = (
                "Error Input. Output exactly one valid 'Action: ...' line in the required format."
            )
        else:
            consecutive_parse_failures = 0
            action_steps += 1
            env_steps += 1
            in_admissible = normalized in admissible
            invalid_action = not in_admissible
            if invalid_action:
                invalid_actions += 1
            if normalized == last_action:
                repeated_actions += 1
            last_action = normalized
            next_obs, reward, done, next_info = env.step([normalized])
            observation = core.normalize_observation(next_obs[0])
            if core.is_invalid_observation(observation):
                nothing_happens += 1
            final_reward = core.scalar_int(reward)
            done_flag = core.scalar_bool(done)
            info_won = next_info.get("won", [False])
            won_flag = bool(
                final_reward > 0
                or core.scalar_bool(info_won)
                or "you won" in observation.lower()
            )
            info = next_info
            admissible = core.current_commands(info)

        messages.append({"role": "user", "content": f"Observation: {observation}"})
        trajectory.append(
            {
                "agent_turn": agent_turns,
                "env_step": env_steps,
                "raw_output": raw_output,
                "normalized": normalized,
                "kind": "parse_failure" if parse_failure_reason else "action",
                "observation": observation,
                "done": done_flag,
                "won": won_flag,
                "reward": final_reward,
                "error": None,
                "parse_failure_reason": parse_failure_reason,
                "in_admissible_commands": in_admissible,
                "invalid_action": invalid_action,
                "invalid_observation": core.is_invalid_observation(observation),
                "consecutive_parse_failures": consecutive_parse_failures,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            }
        )
        if consecutive_parse_failures >= args.max_consecutive_parse_failures:
            termination_reason = "parse_failure_loop"
            break
        if won_flag:
            termination_reason = "success"
            break
        if done_flag:
            termination_reason = "env_done_without_success"
            break
    else:
        if env_steps >= args.max_steps:
            termination_reason = "max_env_steps"

    return {
        "episode_index": index,
        "game_id": game_id,
        "gamefile": gamefile,
        "task_type": task_type,
        "success": won_flag,
        "done": done_flag,
        "reward": final_reward,
        "error": error_message is not None,
        "error_message": error_message,
        "termination_reason": termination_reason,
        "agent_turns": agent_turns,
        "env_steps": env_steps,
        "steps": env_steps,
        "action_steps": action_steps,
        "thought_steps": thought_steps,
        "planning_calls": 0,
        "summary_calls": 0,
        "llm_calls": agent_turns,
        "parse_failures": parse_failures,
        "json_parse_failures": 0,
        "mode_violations": 0,
        "invalid_actions": invalid_actions,
        "nothing_happens": nothing_happens,
        "repeated_actions": repeated_actions,
        "invalid_action_rate": invalid_actions / action_steps if action_steps else 0.0,
        "parse_failure_rate": parse_failures / agent_turns if agent_turns else 0.0,
        "repeated_action_rate": repeated_actions / action_steps if action_steps else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "decision_prompt_tokens": prompt_tokens,
        "decision_completion_tokens": completion_tokens,
        "planning_prompt_tokens": 0,
        "planning_completion_tokens": 0,
        "summary_prompt_tokens": 0,
        "summary_completion_tokens": 0,
        "elapsed_sec": time.perf_counter() - started,
        "initial_plan": None,
        "final_memory_summary": "",
        "summarized_items": 0,
        "selected_prompt_keys": list(keys_by_task[task_type]),
        "initial_observation": initial_obs,
        "trajectory": trajectory,
    }


def evaluate_slice(
    worker_id: int, gamefiles: list[str], args: ProbeArgs
) -> list[dict[str, Any]]:
    config = core.load_yaml(ROOT / args.config)
    demos, keys = load_demo_messages(ROOT / args.prompt_file, args.strategy)
    env = core.make_env(config, "eval_out_of_distribution", gamefiles)
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        http_client=httpx.Client(timeout=args.timeout, trust_env=args.trust_env),
    )
    rows = []
    allowed_gamefiles = {str(Path(path).resolve()) for path in gamefiles}
    for offset, gamefile in enumerate(gamefiles):
        row = run_episode(
            env,
            client,
            args,
            demos,
            keys,
            worker_id * 1_000_000 + offset,
            gamefile,
            allowed_gamefiles,
        )
        rows.append(row)
        print(
            f"worker={worker_id} {offset + 1}/{len(gamefiles)} "
            f"success={int(row['success'])} task={row['task_type']} "
            f"steps={row['env_steps']} turns={row['agent_turns']} "
            f"parse={row['parse_failures']} termination={row['termination_reason']}",
            flush=True,
        )
    return rows


def summary_args(args: ProbeArgs) -> argparse.Namespace:
    return argparse.Namespace(
        split="eval_out_of_distribution",
        model=args.model,
        base_url=args.base_url,
        prompt_file="prompts/alfworld_3prompts.json",
        agent_mode="react",
        fewshot_strategy=args.strategy,
        examples_per_task=2,
        prompt_seed=42,
        include_admissible_actions=False,
        invalid_action_feedback=False,
        output_format="free",
        guided_decoding_backend=None,
        history_mode="chat_full",
        history_k=10,
        summary_max_tokens=160,
        seed=args.seed,
        gamefile_manifest=args.manifest,
        task_type_filter=None,
        lookat_lamp_action_repair=False,
        lookat_lamp_grammar_hint=False,
        max_steps=args.max_steps,
        max_agent_turns=args.max_agent_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_context_chars=0,
        max_consecutive_parse_failures=args.max_consecutive_parse_failures,
        max_consecutive_thoughts=0,
        workers=args.workers,
        limit=None,
    )


def main() -> None:
    args = parse_args()
    core.ensure_data_dir()
    output_dir = ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    gamefiles = load_probe_gamefiles(ROOT / args.manifest)
    chunks = core.split_evenly(gamefiles, args.workers)
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(evaluate_slice, worker_id, chunk, args): worker_id
            for worker_id, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: row["gamefile"])
    for index, row in enumerate(rows):
        row["episode_index"] = index
    if Counter(row["gamefile"] for row in rows) != Counter(gamefiles):
        raise RuntimeError("Probe output gamefile coverage mismatch")

    summary = core.summarize(rows, summary_args(args))
    summary["wall_time_sec"] = time.perf_counter() - started
    summary["config"].update(
        {
            "prompt_file": args.prompt_file,
            "selected_prompt_keys": {
                task_type: list(selected_keys(args.strategy, task_type))
                for task_type in core.TASK_ORDER
            },
            "actual_examples_per_task": 2 if args.strategy == "matched" else 6,
            "protocol": "mpo_reference_chat_reconstructed",
            "reference_commit": REFERENCE_COMMIT,
            "prompt_representation": "role-structured chat messages",
            "turn_contract": "optional Thought plus exactly one Action in the same response",
            "stop": STOP_WORDS,
            "probe_only": True,
        }
    )
    core.write_outputs(rows, summary, output_dir)
    core.print_summary(summary)
    print(f"\nWrote reference baseline probe to {output_dir}")


if __name__ == "__main__":
    main()
