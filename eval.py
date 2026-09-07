#!/usr/bin/env python3
"""Run the ReAct baseline on ALFWorld text tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from openai import OpenAI


TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

PROMPT_PREFIXES = {
    "pick_and_place_simple": "put",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "examine",
    "pick_two_obj_and_place": "puttwo",
}

TASK_ORDER = [
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
]

COMMAND_VERBS = (
    "go back to",
    "go to",
    "open",
    "close",
    "take",
    "pick up",
    "pick",
    "move",
    "put",
    "place",
    "clean",
    "heat",
    "cool",
    "use",
    "turn on",
    "switch on",
    "look at",
    "examine",
)

ACTION_PATTERNS = [
    r"go to [a-z]+ \d+",
    r"(?:open|close|use) [a-z]+ \d+",
    r"take [a-z]+ \d+(?: from [a-z]+ \d+)?",
    r"move [a-z]+ \d+ to [a-z]+ \d+",
    r"put [a-z]+ \d+ into [a-z]+ \d+",
    r"(?:clean|heat|cool) [a-z]+ \d+ with [a-z]+ \d+",
    r"(?:look at|examine) [a-z]+ \d+(?: with [a-z]+ \d+)?",
    r"inventory",
    r"look",
]

JSON_OUTPUT_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "minLength": 1},
                "action": {"type": "string", "enum": [""]},
            },
            "required": ["thought", "action"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "enum": [""]},
                "action": {"type": "string", "minLength": 1},
            },
            "required": ["thought", "action"],
            "additionalProperties": False,
        },
    ],
}

INVALID_ACTION_FEEDBACK = (
    "Nothing happens. The previous action was invalid in the current state. "
    "Do not repeat it; choose a different valid action consistent with location, "
    "inventory, object identity, and container preconditions."
)


@dataclass(frozen=True)
class PromptBundle:
    text: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class EvalArgs:
    config: str
    prompt_file: str
    split: str
    output_dir: str
    base_url: str
    model: str
    api_key: str
    max_steps: int
    max_agent_turns: int
    max_tokens: int
    temperature: float
    max_context_chars: int
    max_consecutive_parse_failures: int
    max_consecutive_thoughts: int
    workers: int
    limit: int | None
    timeout: float
    trust_env: bool
    examples_per_task: int
    agent_mode: str
    fewshot_strategy: str
    prompt_seed: int
    include_admissible_actions: bool
    invalid_action_feedback: bool
    output_format: str
    guided_decoding_backend: str
    history_mode: str
    history_k: int
    summary_max_tokens: int
    seed: int
    gamefile_manifest: str | None
    task_type_filter: str | None
    lookat_lamp_action_repair: bool
    lookat_lamp_grammar_hint: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--prompt-file", default="prompts/alfworld_3prompts.json")
    parser.add_argument(
        "--split",
        default="eval_out_of_distribution",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
    )
    parser.add_argument("--output-dir", default="results/task3_react_baseline")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum number of ALFWorld env.step calls. ReAct think turns do not count.",
    )
    parser.add_argument(
        "--max-agent-turns",
        type=int,
        default=None,
        help="Maximum LLM calls per episode. Defaults to 3 * max_steps to stop thought/parse loops.",
    )
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=24000,
        help="Character budget for prompt construction. This is not a token budget.",
    )
    parser.add_argument(
        "--max-consecutive-parse-failures",
        type=int,
        default=8,
        help="Stop an episode after this many consecutive parse failures. This prevents deterministic parse loops.",
    )
    parser.add_argument(
        "--max-consecutive-thoughts",
        type=int,
        default=24,
        help="Stop an episode after this many consecutive thought turns without an env action.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional small-run limit for smoke tests.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="Honor proxy environment variables for HTTP requests. Disabled by default for local serving.",
    )
    parser.add_argument("--examples-per-task", type=int, default=2)
    parser.add_argument(
        "--agent-mode",
        choices=["react", "act", "cot_then_act"],
        default="react",
        help="Reasoning schedule used by the agent.",
    )
    parser.add_argument(
        "--fewshot-strategy",
        choices=[
            "matched",
            "random_nonmatching",
            "mixed6",
            "mixed6_target_last",
            "zero",
        ],
        default="matched",
    )
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--include-admissible-actions", action="store_true")
    parser.add_argument("--invalid-action-feedback", action="store_true")
    parser.add_argument("--output-format", choices=["free", "json"], default="free")
    parser.add_argument("--guided-decoding-backend", default="xgrammar")
    parser.add_argument("--history-mode", choices=["full", "recent", "summary"], default="full")
    parser.add_argument("--history-k", type=int, default=10)
    parser.add_argument("--summary-max-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gamefile-manifest",
        default=None,
        help="JSON manifest that fixes the evaluated gamefile set and verifies its SHA-256.",
    )
    parser.add_argument(
        "--task-type-filter",
        choices=TASK_ORDER,
        default=None,
        help="Only evaluate gamefiles whose inferred task type matches this value.",
    )
    parser.add_argument(
        "--lookat-lamp-action-repair",
        action="store_true",
        help=(
            "For look_at_obj_in_light only, rewrite supported examine/look-at-with-lamp "
            "actions to use DESKLAMP after the target object has been successfully taken."
        ),
    )
    parser.add_argument(
        "--lookat-lamp-grammar-hint",
        action="store_true",
        help="Add a look_at_obj_in_light prompt hint about taking the target before using the lamp.",
    )
    return parser.parse_args()


def to_eval_args(args: argparse.Namespace) -> EvalArgs:
    max_agent_turns = args.max_agent_turns if args.max_agent_turns is not None else args.max_steps * 3
    if max_agent_turns < args.max_steps:
        raise ValueError("--max-agent-turns must be at least --max-steps")
    if args.examples_per_task < 0:
        raise ValueError("--examples-per-task must be non-negative")
    if args.history_k < 1:
        raise ValueError("--history-k must be at least 1")
    if args.summary_max_tokens < 1:
        raise ValueError("--summary-max-tokens must be at least 1")
    return EvalArgs(
        config=args.config,
        prompt_file=args.prompt_file,
        split=args.split,
        output_dir=args.output_dir,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_steps=args.max_steps,
        max_agent_turns=max_agent_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_context_chars=args.max_context_chars,
        max_consecutive_parse_failures=max(1, args.max_consecutive_parse_failures),
        max_consecutive_thoughts=max(1, args.max_consecutive_thoughts),
        workers=max(1, args.workers),
        limit=args.limit,
        timeout=args.timeout,
        trust_env=args.trust_env,
        examples_per_task=args.examples_per_task,
        agent_mode=args.agent_mode,
        fewshot_strategy=args.fewshot_strategy,
        prompt_seed=args.prompt_seed,
        include_admissible_actions=args.include_admissible_actions,
        invalid_action_feedback=args.invalid_action_feedback,
        output_format=args.output_format,
        guided_decoding_backend=args.guided_decoding_backend,
        history_mode=args.history_mode,
        history_k=args.history_k,
        summary_max_tokens=args.summary_max_tokens,
        seed=args.seed,
        gamefile_manifest=args.gamefile_manifest,
        task_type_filter=args.task_type_filter,
        lookat_lamp_action_repair=args.lookat_lamp_action_repair,
        lookat_lamp_grammar_hint=args.lookat_lamp_grammar_hint,
    )


def ensure_data_dir() -> str:
    if os.environ.get("ALFWORLD_DATA"):
        return os.environ["ALFWORLD_DATA"]
    default_path = Path("/root/.cache/alfworld")
    if default_path.exists():
        os.environ["ALFWORLD_DATA"] = str(default_path)
        return str(default_path)
    path = Path.home() / ".cache" / "alfworld"
    os.environ["ALFWORLD_DATA"] = str(path)
    return str(path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt_source(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError("Prompt file must contain a JSON object")
    return {str(key): str(value) for key, value in raw.items()}


def transform_example_to_json(example: str) -> str:
    transformed = []
    for line in example.splitlines():
        if not line.startswith("> "):
            transformed.append(line)
            continue
        turn = line[2:].strip()
        if re.match(r"^(?:think|thought)\s*:", turn, flags=re.IGNORECASE):
            thought = re.sub(
                r"^(?:think|thought)\s*:\s*",
                "",
                turn,
                flags=re.IGNORECASE,
            )
            payload = {"thought": thought, "action": ""}
        else:
            payload = {"thought": "", "action": turn}
        transformed.append("> " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(transformed) + ("\n" if example.endswith("\n") else "")


def prompt_keys_for_task(raw: dict[str, str], args: EvalArgs, task_type: str) -> tuple[str, ...]:
    family = "react" if args.agent_mode == "react" else "act"
    prefix = PROMPT_PREFIXES[task_type]

    if args.fewshot_strategy == "zero":
        keys: list[str] = []
    elif args.fewshot_strategy == "matched":
        order = [1, 0] + list(range(2, args.examples_per_task))
        keys = [f"{family}_{prefix}_{index}" for index in order[: args.examples_per_task]]
    elif args.fewshot_strategy in {"mixed6", "mixed6_target_last"}:
        task_order = list(TASK_ORDER)
        if args.fewshot_strategy == "mixed6_target_last":
            task_order.remove(task_type)
            task_order.append(task_type)
        keys = [f"{family}_{PROMPT_PREFIXES[name]}_1" for name in task_order]
    elif args.fewshot_strategy == "random_nonmatching":
        candidates = [
            f"{family}_{other_prefix}_{index}"
            for other_task, other_prefix in PROMPT_PREFIXES.items()
            if other_task != task_type
            for index in range(3)
        ]
        rng = random.Random(f"{args.prompt_seed}:{task_type}:{family}")
        keys = rng.sample(sorted(candidates), 2)
    else:
        raise ValueError(f"Unknown few-shot strategy: {args.fewshot_strategy}")

    missing = [key for key in keys if key not in raw]
    if missing:
        raise KeyError(f"Missing prompt examples: {missing}")
    return tuple(keys)


def load_prompts(path: str | Path, args: EvalArgs) -> dict[str, PromptBundle]:
    raw = load_prompt_source(path)
    prompts: dict[str, PromptBundle] = {}
    for task_type in PROMPT_PREFIXES:
        keys = prompt_keys_for_task(raw, args, task_type)
        examples = [raw[key] for key in keys]
        if args.output_format == "json":
            examples = [transform_example_to_json(example) for example in examples]
        prompts[task_type] = PromptBundle(text="".join(examples).rstrip(), keys=keys)
    return prompts


def get_env_class(env_type: str):
    import alfworld.agents.environment as environment

    env_class = getattr(environment, env_type, None)
    if env_class is None and hasattr(environment, "get_environment"):
        env_class = environment.get_environment(env_type)
    if env_class is None:
        raise ValueError(f"Unknown ALFWorld environment type: {env_type}")
    return env_class


def make_collector(config: dict[str, Any], split: str):
    env_class = get_env_class(config["env"]["type"])
    return env_class(config, train_eval=split)


def make_env(config: dict[str, Any], split: str, game_files: list[str] | None = None):
    collector = make_collector(config, split)
    if game_files is not None:
        collector.game_files = list(game_files)
        collector.num_games = len(collector.game_files)
    return collector.init_env(batch_size=1)


def get_task_type_from_gamefile(gamefile: str) -> str:
    path = Path(gamefile)
    task_dir = path.parent.parent.name if path.name == "game.tw-pddl" else path.parent.name
    for task_type in PROMPT_PREFIXES:
        if task_dir.startswith(task_type):
            return task_type
    if task_dir.startswith("look_at_obj"):
        return "look_at_obj_in_light"
    if task_dir.startswith("pick_two_obj"):
        return "pick_two_obj_and_place"
    raise ValueError(f"Cannot infer task type from gamefile: {gamefile}")


def game_id_from_gamefile(gamefile: str) -> str:
    path = Path(gamefile)
    if path.name == "game.tw-pddl":
        return "/".join(path.parts[-3:-1])
    return "/".join(path.parts[-2:])


def normalize_observation(obs: str) -> str:
    obs = obs.replace("\r\n", "\n").strip()
    obs = re.sub(r"\n{3,}", "\n\n", obs)
    if obs.startswith("You arrive at loc "):
        pos = obs.find(". ")
        if pos >= 0:
            obs = obs[pos + 2 :]
    return obs


def normalize_initial_observation(obs: str) -> str:
    obs = normalize_observation(obs)
    parts = obs.split("\n\n")
    if parts and parts[0].startswith("-= Welcome to TextWorld"):
        obs = "\n\n".join(parts[1:]).strip()
    return obs


def clean_model_text(text: str) -> str:
    text = text.replace("\r", "\n").strip()
    if not text:
        return ""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    # Some completion models continue the transcript on the same line, e.g.
    # "I will retry. > think: ...". Keep only the first generated turn.
    first_line = re.split(r"\s+>\s+", first_line, maxsplit=1)[0]
    return first_line.strip().strip("` ")


def strip_articles(text: str) -> str:
    return re.sub(r"\b(?:the|a|an)\s+(?=[a-z]+\s+\d+\b)", "", text)


def canonicalize_candidate(text: str) -> str:
    action = text.strip().strip('"\'` ')
    action = re.sub(r"\s+", " ", action).lower()
    action = strip_articles(action)
    action = re.sub(r"^go back to ", "go to ", action)
    action = re.sub(r"^pick up ", "take ", action)
    action = re.sub(r"^pick ", "take ", action)
    action = re.sub(r"^(?:turn on|switch on) ([a-z]+ \d+).*$", r"use \1", action)
    action = re.sub(r"^place ", "put ", action)
    action = re.sub(r"^move ([a-z]+ \d+) (?:in|on|into|onto) ([a-z]+ \d+)$", r"move \1 to \2", action)
    use_match = re.match(r"^use ([a-z]+ \d+)(?:\s+to\b.*)?$", action)
    if use_match:
        action = f"use {use_match.group(1)}"
    put_inon_match = re.match(r"^put ([a-z]+ \d+) in/on ([a-z]+ \d+)$", action)
    if put_inon_match:
        action = f"move {put_inon_match.group(1)} to {put_inon_match.group(2)}"
    else:
        put_recep_match = re.match(r"^put ([a-z]+ \d+) (?:in|on|onto) ([a-z]+ \d+)$", action)
        if put_recep_match:
            action = f"move {put_recep_match.group(1)} to {put_recep_match.group(2)}"
    return re.sub(r"\s+", " ", action).strip()


def looks_like_action(action: str) -> bool:
    return any(re.fullmatch(pattern, action, flags=re.IGNORECASE) for pattern in ACTION_PATTERNS)


def action_segments(text: str) -> list[str]:
    lower = text.lower()
    verb_pattern = r"\b(?:" + "|".join(re.escape(verb) for verb in COMMAND_VERBS) + r")\b"
    segments = []
    for match in re.finditer(verb_pattern, lower):
        prefix = lower[max(0, match.start() - 20) : match.start()]
        if re.search(r"(?:can\s*not|cannot|can't|cant|don't|do not|already)\s+$", prefix):
            continue
        segment = lower[match.start() :]
        segment = re.split(
            r"(?:[.;]|,\s*(?:but|so|and)|\s+because\s+|\s+as\s+|\s+and\s+then\s+|\s+then\s+)",
            segment,
            maxsplit=1,
        )[0]
        segments.append(segment.strip())
    return segments


def looks_like_unprefixed_thought(text: str) -> bool:
    candidate = text.strip().strip('"` ').lower()
    candidate = re.sub(r"^>\s*", "", candidate).strip()
    if not candidate:
        return False
    if candidate in {"ouch", "ouch!"}:
        return False
    if re.fullmatch(r"[>\s.。!！?？]+", candidate):
        return False
    if re.fullmatch(r"\d+\.?", candidate):
        return False
    action_like = re.match(
        r"^(?:go(?: back)? to|open|close|take|pick(?: up)?|move|put|place|clean|heat|cool|use|turn on|switch on|look at|examine|inventory|look)\b",
        candidate,
    )
    if action_like:
        return False
    thought_starts = (
        "i ",
        "i'",
        "i’ll",
        "i’ll",
        "now i",
        "let me",
        "we ",
        "the task",
        "it seems",
        "there ",
        "this ",
        "that ",
        "you ",
        "ouch:",
        "ouch,",
        "ouch! ",
    )
    thought_words = (
        "need",
        "should",
        "will",
        "found",
        "find",
        "already",
        "cannot",
        "can't",
        "not",
        "completed",
        "checked",
        "try",
        "seems",
        "available",
        "possible",
        "task",
    )
    return candidate.startswith(thought_starts) or any(re.search(rf"\b{re.escape(word)}\b", candidate) for word in thought_words)


def normalize_action(text: str) -> tuple[str, str]:
    raw = clean_model_text(text)
    action = raw

    action = re.sub(r"^(?:action|act)\s*[:：]\s*", "", action, flags=re.IGNORECASE)
    action = re.sub(r"^\s*\d+[\.)]\s*", "", action)
    action = re.sub(r"^(?:>\s*)", "", action).strip()
    action = action.strip().strip('"\'` ')
    action = re.sub(r"\s+", " ", action)

    if re.match(r"^(?:think|thought)\s*[:：]", action, flags=re.IGNORECASE):
        thought = re.sub(r"^(?:think|thought)\s*[:：]\s*", "think: ", action, flags=re.IGNORECASE)
        return thought.strip(), "thought"

    action = action.rstrip(".。")
    direct = canonicalize_candidate(action)
    if looks_like_action(direct):
        return direct, "action"

    for segment in action_segments(action):
        candidate = canonicalize_candidate(segment)
        if looks_like_action(candidate):
            return candidate, "action"

    if looks_like_unprefixed_thought(action):
        thought = re.sub(r"^>\s*", "", action).strip()
        return f"think: {thought}", "thought"

    if not direct:
        return raw, "parse_failure"
    return direct, "parse_failure"


def parse_model_output(text: str, output_format: str) -> tuple[str, str, str | None]:
    if output_format == "free":
        normalized, kind = normalize_action(text)
        return normalized, kind, None

    raw = text.strip().strip("\x60 ")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, "parse_failure", "json_syntax"
    if not isinstance(payload, dict) or set(payload) != {"thought", "action"}:
        return raw, "parse_failure", "json_schema"
    thought = payload.get("thought")
    action = payload.get("action")
    if not isinstance(thought, str) or not isinstance(action, str):
        return raw, "parse_failure", "json_schema"
    thought = thought.strip()
    action = action.strip()
    if bool(thought) == bool(action):
        return raw, "parse_failure", "json_exclusive_fields"
    if thought:
        return f"think: {thought}", "thought", None
    normalized, kind = normalize_action(action)
    if kind != "action":
        return normalized, "parse_failure", "json_action_content"
    return normalized, kind, None


def build_prompt(
    bundle: PromptBundle,
    initial_obs: str,
    args: EvalArgs,
    initial_plan: str | None = None,
    task_type: str | None = None,
) -> str:
    count = len(bundle.keys)
    if count == 2:
        intro = "Interact with a household to solve a task. Here are two examples."
    elif count:
        intro = f"Interact with a household to solve a task. Here are {count} examples."
    else:
        intro = "Interact with a household to solve a task."

    instructions = []
    if args.agent_mode in {"act", "cot_then_act"}:
        instructions.append("Output exactly one environment action per turn. Do not output thoughts.")
    if args.output_format == "json":
        instructions.append(
            'Output one JSON object per turn with exactly the keys "thought" and "action". '
            "Exactly one value must be non-empty."
        )
    if args.lookat_lamp_grammar_hint and task_type == "look_at_obj_in_light":
        instructions.append(
            'For look_at_obj_in_light tasks, do not output "examine X with desklamp". '
            "First take X, then use desklamp N while holding X."
        )

    sections = [intro]
    sections.extend(instructions)
    if bundle.text:
        sections.append(bundle.text.strip())
    sections.extend(["Here is the task.", normalize_initial_observation(initial_obs)])
    if initial_plan:
        sections.append(f"Initial plan (do not rewrite it): {initial_plan.strip()}")
    return "\n".join(sections)


def build_admissible_suffix(commands: list[str]) -> str:
    encoded = json.dumps(sorted(commands), ensure_ascii=False, separators=(",", ":"))
    return (
        f"Current admissible actions: {encoded}\n"
        "If acting, choose exactly one action from this current list."
    )


def build_context(
    base_prompt: str,
    trajectory_text: list[str],
    max_context_chars: int,
    history_mode: str = "full",
    history_k: int = 10,
    memory_summary: str = "",
    summarized_items: int = 0,
    dynamic_suffix: str = "",
) -> tuple[str, dict[str, Any]]:
    total = len(trajectory_text)
    if history_mode == "full":
        candidates = trajectory_text
        policy_dropped = 0
    elif history_mode == "recent":
        candidates = trajectory_text[-history_k:]
        policy_dropped = max(0, total - len(candidates))
    elif history_mode == "summary":
        candidates = trajectory_text[summarized_items:]
        policy_dropped = summarized_items
    else:
        raise ValueError(f"Unknown history mode: {history_mode}")

    fixed_sections = [base_prompt]
    if memory_summary:
        fixed_sections.append(f"Memory summary: {memory_summary.strip()}")
    fixed = "\n".join(fixed_sections)
    reserve = len(dynamic_suffix) + (1 if dynamic_suffix else 0) + 3
    if len(fixed) + reserve > max_context_chars:
        raise ValueError("Base prompt, memory, and dynamic suffix exceed --max-context-chars")

    suffix = ""
    kept = 0
    for item in reversed(candidates):
        candidate = item + ("\n" + suffix if suffix else "")
        if len(fixed) + 1 + len(candidate) + reserve > max_context_chars and kept > 0:
            break
        suffix = candidate
        kept += 1
        if len(fixed) + 1 + len(suffix) + reserve > max_context_chars:
            break

    sections = [fixed]
    if suffix:
        sections.append(suffix)
    if dynamic_suffix:
        sections.append(dynamic_suffix)
    context = "\n".join(sections)
    budget_dropped = max(0, len(candidates) - kept)
    meta = {
        "context_chars": len(context),
        "max_context_chars": max_context_chars,
        "history_mode": history_mode,
        "history_items_total": total,
        "history_items_kept": kept,
        "history_items_policy_dropped": policy_dropped,
        "history_items_budget_dropped": budget_dropped,
        "context_truncated": policy_dropped + budget_dropped > 0,
        "context_budget_truncated": budget_dropped > 0,
        "base_prompt_chars": len(base_prompt),
        "memory_summary_chars": len(memory_summary),
        "dynamic_suffix_chars": len(dynamic_suffix),
    }
    return context, meta


def stable_request_seed(base_seed: int, game_id: str, purpose: str, ordinal: int) -> int:
    value = f"{base_seed}:{game_id}:{purpose}:{ordinal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") & 0x7FFFFFFF


def response_usage(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return prompt_tokens, completion_tokens, total_tokens


def llm(
    client: OpenAI,
    args: EvalArgs,
    prompt: str,
    *,
    request_seed: int,
    max_tokens: int | None = None,
    guided_json: bool = False,
) -> LLMResult:
    extra_body = None
    if guided_json:
        extra_body = {
            "guided_json": JSON_OUTPUT_SCHEMA,
            "guided_decoding_backend": args.guided_decoding_backend,
        }
    response = client.completions.create(
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=max_tokens if max_tokens is not None else args.max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None if guided_json else ["\n"],
        seed=request_seed,
        extra_body=extra_body,
    )
    prompt_tokens, completion_tokens, total_tokens = response_usage(response)
    return LLMResult(
        text=response.choices[0].text.strip(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def make_initial_plan(
    client: OpenAI,
    args: EvalArgs,
    game_id: str,
    initial_obs: str,
) -> LLMResult:
    prompt = (
        "Plan an ALFWorld task before acting. Write one concise high-level plan. "
        "Do not invent object IDs or assume the contents of unopened containers. "
        "After this plan, actions will be selected without further explicit reasoning.\n"
        f"{normalize_initial_observation(initial_obs)}\nPlan:"
    )
    return llm(
        client,
        args,
        prompt,
        request_seed=stable_request_seed(args.seed, game_id, "initial_plan", 0),
        max_tokens=args.max_tokens,
    )


def update_memory_summary(
    client: OpenAI,
    args: EvalArgs,
    game_id: str,
    prior_summary: str,
    dropped_turn: str,
    ordinal: int,
) -> LLMResult:
    prompt = (
        "Update the compact memory for an ALFWorld agent. Preserve only verified facts: "
        "visited locations and observed objects, inventory, completed state changes, "
        "placed object IDs, failed actions, and remaining subgoals. Do not invent facts.\n"
        f"Previous memory: {prior_summary or '(empty)'}\n"
        f"New interaction: {dropped_turn}\n"
        "Updated memory:"
    )
    return llm(
        client,
        args,
        prompt,
        request_seed=stable_request_seed(args.seed, game_id, "summary", ordinal),
        max_tokens=args.summary_max_tokens,
    )


def is_invalid_observation(obs: str) -> bool:
    return obs.strip().lower().startswith("nothing happens")


def scalar_bool(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if value else False
    return bool(value)


def scalar_int(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 0
    return int(value)


def current_commands(info: dict[str, Any]) -> list[str]:
    commands = info.get("admissible_commands", [[]]) if isinstance(info, dict) else [[]]
    if isinstance(commands, (list, tuple)) and commands and isinstance(commands[0], (list, tuple)):
        return list(commands[0])
    return list(commands) if isinstance(commands, (list, tuple)) else []


LOOKAT_LAMP_REPAIR_TYPE = "lookat_lamp_examine_to_use"
LOOKAT_LAMP_REPAIR_RE = re.compile(
    r"^(?:examine|look at) (?P<target>[a-z]+ \d+) with (?P<lamp>desklamp \d+)$"
)
TAKE_FROM_RE = re.compile(r"^take (?P<object>[a-z]+ \d+) from [a-z]+ \d+$")
MOVE_TO_RE = re.compile(r"^move (?P<object>[a-z]+ \d+) to [a-z]+ \d+$")


def repair_action(
    normalized_action: str,
    task_type: str,
    lookat_lamp_action_repair: bool,
    held_objects: set[str],
) -> tuple[str, bool, str | None]:
    if not lookat_lamp_action_repair:
        return normalized_action, False, None
    if task_type != "look_at_obj_in_light":
        return normalized_action, False, None
    match = LOOKAT_LAMP_REPAIR_RE.fullmatch(normalized_action)
    if not match:
        return normalized_action, False, None
    target = match.group("target")
    if target not in held_objects:
        return normalized_action, False, None
    return f"use {match.group('lamp')}", True, LOOKAT_LAMP_REPAIR_TYPE


def update_held_objects_after_action(
    action: str,
    invalid_observation: bool,
    held_objects: set[str],
) -> None:
    if invalid_observation:
        return
    take_match = TAKE_FROM_RE.fullmatch(action)
    if take_match:
        held_objects.add(take_match.group("object"))
        return
    move_match = MOVE_TO_RE.fullmatch(action)
    if move_match:
        held_objects.discard(move_match.group("object"))


def empty_episode(
    index: int,
    gamefile: str,
    task_type: str,
    error: str,
    termination_reason: str,
) -> dict[str, Any]:
    return {
        "episode_index": index,
        "game_id": game_id_from_gamefile(gamefile) if gamefile else "unknown",
        "gamefile": gamefile,
        "task_type": task_type,
        "success": False,
        "done": False,
        "reward": 0,
        "error": True,
        "error_message": error,
        "termination_reason": termination_reason,
        "agent_turns": 0,
        "env_steps": 0,
        "steps": 0,
        "action_steps": 0,
        "thought_steps": 0,
        "planning_calls": 0,
        "summary_calls": 0,
        "llm_calls": 0,
        "parse_failures": 0,
        "json_parse_failures": 0,
        "mode_violations": 0,
        "invalid_actions": 0,
        "nothing_happens": 0,
        "repeated_actions": 0,
        "action_repairs": 0,
        "lookat_lamp_repairs": 0,
        "repair_rate": 0.0,
        "invalid_action_rate": 0.0,
        "parse_failure_rate": 0.0,
        "repeated_action_rate": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "decision_prompt_tokens": 0,
        "decision_completion_tokens": 0,
        "planning_prompt_tokens": 0,
        "planning_completion_tokens": 0,
        "summary_prompt_tokens": 0,
        "summary_completion_tokens": 0,
        "elapsed_sec": 0.0,
        "initial_plan": None,
        "final_memory_summary": "",
        "summarized_items": 0,
        "selected_prompt_keys": [],
        "initial_observation": "",
        "trajectory": [],
    }


def history_turn(raw_output: str, normalized: str, kind: str, output_format: str) -> str:
    if output_format == "json" and kind in {"thought", "action"}:
        return raw_output.strip()
    return normalized


def run_episode(
    env: Any,
    prompts: dict[str, PromptBundle],
    args: EvalArgs,
    index: int,
    expected_gamefile: str,
    allowed_gamefiles: set[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_type = get_task_type_from_gamefile(expected_gamefile)
    try:
        obs, info = env.reset()
        gamefile = info["extra.gamefile"][0]
        resolved_gamefile = str(Path(gamefile).resolve())
        if allowed_gamefiles is not None and resolved_gamefile not in allowed_gamefiles:
            raise RuntimeError(f"Unexpected gamefile outside worker manifest: {gamefile}")
        task_type = get_task_type_from_gamefile(gamefile)
        initial_obs = normalize_initial_observation(obs[0])
    except Exception as exc:
        row = empty_episode(index, expected_gamefile, task_type, repr(exc), "reset_error")
        row["elapsed_sec"] = time.perf_counter() - started
        return row

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        http_client=httpx.Client(timeout=args.timeout, trust_env=args.trust_env),
    )
    game_id = game_id_from_gamefile(gamefile)
    bundle = prompts[task_type]
    initial_plan = None
    planning_calls = 0
    planning_prompt_tokens = 0
    planning_completion_tokens = 0
    planning_total_tokens = 0

    if args.agent_mode == "cot_then_act":
        try:
            plan_result = make_initial_plan(client, args, game_id, initial_obs)
            initial_plan = plan_result.text
            planning_calls = 1
            planning_prompt_tokens = plan_result.prompt_tokens
            planning_completion_tokens = plan_result.completion_tokens
            planning_total_tokens = plan_result.total_tokens
        except Exception as exc:
            row = empty_episode(index, gamefile, task_type, repr(exc), "planning_error")
            row["planning_calls"] = 1
            row["llm_calls"] = 1
            row["elapsed_sec"] = time.perf_counter() - started
            row["initial_observation"] = initial_obs
            row["selected_prompt_keys"] = list(bundle.keys)
            return row

    base_prompt = build_prompt(bundle, initial_obs, args, initial_plan, task_type=task_type)
    trajectory_text: list[str] = []
    trajectory: list[dict[str, Any]] = []
    memory_summary = ""
    summarized_items = 0
    summary_calls = 0
    summary_prompt_tokens = 0
    summary_completion_tokens = 0
    summary_total_tokens = 0
    agent_turns = 0
    env_steps = 0
    action_steps = 0
    thought_steps = 0
    parse_failures = 0
    json_parse_failures = 0
    mode_violations = 0
    invalid_actions = 0
    nothing_happens = 0
    repeated_actions = 0
    action_repairs = 0
    lookat_lamp_repairs = 0
    held_objects: set[str] = set()
    decision_prompt_tokens = 0
    decision_completion_tokens = 0
    decision_total_tokens = 0
    consecutive_parse_failures = 0
    consecutive_thoughts = 0
    last_action = None
    done_flag = False
    won_flag = False
    final_reward = 0
    error_flag = False
    error_message = None
    termination_reason = "max_env_steps"
    admissible = current_commands(info)

    while env_steps < args.max_steps and agent_turns < args.max_agent_turns:
        if args.history_mode == "summary":
            try:
                while len(trajectory_text) - summarized_items > args.history_k:
                    summary_result = update_memory_summary(
                        client,
                        args,
                        game_id,
                        memory_summary,
                        trajectory_text[summarized_items],
                        summary_calls,
                    )
                    memory_summary = summary_result.text
                    summarized_items += 1
                    summary_calls += 1
                    summary_prompt_tokens += summary_result.prompt_tokens
                    summary_completion_tokens += summary_result.completion_tokens
                    summary_total_tokens += summary_result.total_tokens
            except Exception as exc:
                error_flag = True
                error_message = repr(exc)
                termination_reason = "summary_error"
                break

        dynamic_suffix = (
            build_admissible_suffix(admissible) if args.include_admissible_actions else ""
        )
        try:
            context, context_meta = build_context(
                base_prompt,
                trajectory_text,
                args.max_context_chars,
                history_mode=args.history_mode,
                history_k=args.history_k,
                memory_summary=memory_summary,
                summarized_items=summarized_items,
                dynamic_suffix=dynamic_suffix,
            )
        except Exception as exc:
            error_flag = True
            error_message = repr(exc)
            termination_reason = "context_error"
            break

        prompt = f"{context}\n> "
        agent_turns += 1
        raw_output = ""
        normalized = ""
        raw_intended_action = ""
        normalized_before_repair = ""
        normalized_after_repair = ""
        repair_applied = False
        repair_type = None
        kind = "error"
        observation = ""
        model_feedback = ""
        step_error = None
        in_admissible = None
        invalid_action = False
        invalid_observation = False
        parse_failure_reason = None
        turn_prompt_tokens = 0
        turn_completion_tokens = 0
        turn_total_tokens = 0

        try:
            result = llm(
                client,
                args,
                prompt,
                request_seed=stable_request_seed(args.seed, game_id, "decision", agent_turns),
                guided_json=args.output_format == "json",
            )
            raw_output = result.text
            turn_prompt_tokens = result.prompt_tokens
            turn_completion_tokens = result.completion_tokens
            turn_total_tokens = result.total_tokens
            decision_prompt_tokens += result.prompt_tokens
            decision_completion_tokens += result.completion_tokens
            decision_total_tokens += result.total_tokens
            normalized, kind, parse_failure_reason = parse_model_output(
                raw_output, args.output_format
            )
            raw_intended_action = raw_output
            normalized_before_repair = normalized
            normalized_after_repair = normalized

            if kind == "thought" and args.agent_mode != "react":
                kind = "parse_failure"
                parse_failure_reason = "mode_violation"
                mode_violations += 1

            if kind == "thought":
                thought_steps += 1
                consecutive_parse_failures = 0
                consecutive_thoughts += 1
                observation = "OK."
                model_feedback = observation

            elif kind == "parse_failure":
                parse_failures += 1
                if parse_failure_reason and parse_failure_reason.startswith("json_"):
                    json_parse_failures += 1
                consecutive_parse_failures += 1
                consecutive_thoughts = 0
                observation = "Nothing happens."
                model_feedback = observation

            elif kind == "action":
                normalized, repair_applied, repair_type = repair_action(
                    normalized,
                    task_type,
                    args.lookat_lamp_action_repair,
                    held_objects,
                )
                normalized_after_repair = normalized
                if repair_applied:
                    action_repairs += 1
                    if repair_type == LOOKAT_LAMP_REPAIR_TYPE:
                        lookat_lamp_repairs += 1

                action_steps += 1
                consecutive_parse_failures = 0
                consecutive_thoughts = 0
                env_steps += 1
                in_admissible = normalized in admissible
                if normalized == last_action:
                    repeated_actions += 1
                last_action = normalized

                next_obs, reward, done, next_info = env.step([normalized])
                observation = normalize_observation(next_obs[0])
                invalid_observation = is_invalid_observation(observation)
                if invalid_observation:
                    nothing_happens += 1
                update_held_objects_after_action(normalized, invalid_observation, held_objects)
                model_feedback = (
                    INVALID_ACTION_FEEDBACK
                    if args.invalid_action_feedback and invalid_observation
                    else observation
                )
                final_reward = scalar_int(reward)
                done_flag = scalar_bool(done)
                info_won = next_info.get("won", [False]) if isinstance(next_info, dict) else [False]
                won_flag = bool(
                    final_reward > 0
                    or scalar_bool(info_won)
                    or "you won" in observation.lower()
                )

                invalid_action = not in_admissible
                if invalid_action:
                    invalid_actions += 1
                info = next_info
                admissible = current_commands(info)

            else:
                parse_failures += 1
                consecutive_parse_failures += 1
                consecutive_thoughts = 0
                observation = "Nothing happens."
                model_feedback = observation

            rendered_turn = history_turn(
                raw_output, normalized, kind, args.output_format
            )
            trajectory_text.append(f"> {rendered_turn}\n{model_feedback}")

        except Exception as exc:
            error_flag = True
            step_error = repr(exc)
            error_message = step_error
            observation = f"ERROR: {step_error}"
            model_feedback = observation
            termination_reason = "runtime_error"
            trajectory.append(
                {
                    "agent_turn": agent_turns,
                    "env_step": env_steps,
                    "raw_output": raw_output,
                    "normalized": normalized,
                    "raw_intended_action": raw_intended_action,
                    "normalized_before_repair": normalized_before_repair,
                    "normalized_after_repair": normalized_after_repair,
                    "repair_applied": repair_applied,
                    "repair_type": repair_type,
                    "kind": kind,
                    "observation": observation,
                    "model_feedback": model_feedback,
                    "parse_failure_reason": parse_failure_reason,
                    "prompt_tokens": turn_prompt_tokens,
                    "completion_tokens": turn_completion_tokens,
                    "total_tokens": turn_total_tokens,
                    "error": step_error,
                    **context_meta,
                }
            )
            break

        trajectory.append(
            {
                "agent_turn": agent_turns,
                "env_step": env_steps,
                "raw_output": raw_output,
                "normalized": normalized,
                "raw_intended_action": raw_intended_action,
                "normalized_before_repair": normalized_before_repair,
                "normalized_after_repair": normalized_after_repair,
                "repair_applied": repair_applied,
                "repair_type": repair_type,
                "kind": kind,
                "observation": observation,
                "model_feedback": model_feedback,
                "done": done_flag,
                "won": won_flag,
                "reward": final_reward,
                "error": step_error,
                "parse_failure_reason": parse_failure_reason,
                "in_admissible_commands": in_admissible,
                "invalid_action": invalid_action,
                "invalid_observation": invalid_observation,
                "consecutive_parse_failures": consecutive_parse_failures,
                "consecutive_thoughts": consecutive_thoughts,
                "prompt_tokens": turn_prompt_tokens,
                "completion_tokens": turn_completion_tokens,
                "total_tokens": turn_total_tokens,
                **context_meta,
            }
        )

        if consecutive_parse_failures >= args.max_consecutive_parse_failures:
            termination_reason = "parse_failure_loop"
            break
        if consecutive_thoughts >= args.max_consecutive_thoughts:
            termination_reason = "thought_loop"
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
        elif agent_turns >= args.max_agent_turns:
            termination_reason = "max_agent_turns"

    prompt_tokens = (
        decision_prompt_tokens + planning_prompt_tokens + summary_prompt_tokens
    )
    completion_tokens = (
        decision_completion_tokens
        + planning_completion_tokens
        + summary_completion_tokens
    )
    total_tokens = decision_total_tokens + planning_total_tokens + summary_total_tokens
    elapsed = time.perf_counter() - started
    return {
        "episode_index": index,
        "game_id": game_id,
        "gamefile": gamefile,
        "task_type": task_type,
        "success": won_flag,
        "done": done_flag,
        "reward": final_reward,
        "error": error_flag,
        "error_message": error_message,
        "termination_reason": termination_reason,
        "agent_turns": agent_turns,
        "env_steps": env_steps,
        "steps": env_steps,
        "action_steps": action_steps,
        "thought_steps": thought_steps,
        "planning_calls": planning_calls,
        "summary_calls": summary_calls,
        "llm_calls": agent_turns + planning_calls + summary_calls,
        "parse_failures": parse_failures,
        "json_parse_failures": json_parse_failures,
        "mode_violations": mode_violations,
        "invalid_actions": invalid_actions,
        "nothing_happens": nothing_happens,
        "repeated_actions": repeated_actions,
        "action_repairs": action_repairs,
        "lookat_lamp_repairs": lookat_lamp_repairs,
        "repair_rate": action_repairs / action_steps if action_steps else 0.0,
        "invalid_action_rate": invalid_actions / action_steps if action_steps else 0.0,
        "parse_failure_rate": parse_failures / agent_turns if agent_turns else 0.0,
        "repeated_action_rate": repeated_actions / action_steps if action_steps else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "decision_prompt_tokens": decision_prompt_tokens,
        "decision_completion_tokens": decision_completion_tokens,
        "planning_prompt_tokens": planning_prompt_tokens,
        "planning_completion_tokens": planning_completion_tokens,
        "summary_prompt_tokens": summary_prompt_tokens,
        "summary_completion_tokens": summary_completion_tokens,
        "elapsed_sec": elapsed,
        "initial_plan": initial_plan,
        "final_memory_summary": memory_summary,
        "summarized_items": summarized_items,
        "selected_prompt_keys": list(bundle.keys),
        "initial_observation": initial_obs,
        "trajectory": trajectory,
    }


def evaluate_slice(worker_id: int, game_files: list[str], args: EvalArgs) -> list[dict[str, Any]]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed + worker_id)
    ensure_data_dir()
    try:
        config = load_yaml(args.config)
        prompts = load_prompts(args.prompt_file, args)
        env = make_env(config, args.split, game_files)
    except Exception as exc:
        return [
            empty_episode(
                worker_id * 1_000_000 + offset,
                gamefile,
                get_task_type_from_gamefile(gamefile),
                repr(exc),
                "worker_init_error",
            )
            for offset, gamefile in enumerate(game_files)
        ]

    rows = []
    allowed_gamefiles = {str(Path(path).resolve()) for path in game_files}
    for offset, gamefile in enumerate(game_files):
        episode_index = worker_id * 1_000_000 + offset
        row = run_episode(
            env, prompts, args, episode_index, gamefile, allowed_gamefiles
        )
        rows.append(row)
        print(
            f"worker {worker_id}: {offset + 1}/{len(game_files)} "
            f"success={int(row['success'])} task={row['task_type']} "
            f"env_steps={row['env_steps']} agent_turns={row['agent_turns']} "
            f"thoughts={row['thought_steps']} parse_failures={row['parse_failures']} "
            f"llm_calls={row['llm_calls']} tokens={row['total_tokens']} "
            f"termination={row['termination_reason']}",
            flush=True,
        )

    observed_gamefiles = Counter(
        str(Path(row["gamefile"]).resolve()) for row in rows if row.get("gamefile")
    )
    expected_gamefiles = Counter(allowed_gamefiles)
    if observed_gamefiles != expected_gamefiles:
        missing = list((expected_gamefiles - observed_gamefiles).elements())
        unexpected = list((observed_gamefiles - expected_gamefiles).elements())
        raise RuntimeError(
            f"Worker {worker_id} game coverage mismatch: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    return rows


def gamefile_set_sha256(game_files: list[str]) -> str:
    payload = "\n".join(sorted(game_files)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_gamefile_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "gamefiles" in manifest:
        game_files = [str(item) for item in manifest["gamefiles"]]
    elif "source_trajectories" in manifest:
        source = Path(manifest["source_trajectories"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        with source.open("r", encoding="utf-8") as f:
            game_files = [json.loads(line)["gamefile"] for line in f if line.strip()]
    else:
        raise KeyError("Manifest requires gamefiles or source_trajectories")

    game_files = sorted(game_files)
    expected_count = int(manifest.get("expected_count", len(game_files)))
    if len(game_files) != expected_count:
        raise ValueError(
            f"Manifest expected {expected_count} gamefiles, found {len(game_files)}"
        )
    if len(set(game_files)) != len(game_files):
        raise ValueError("Manifest contains duplicate gamefiles")
    expected_hash = manifest.get("gamefile_set_sha256")
    actual_hash = gamefile_set_sha256(game_files)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Manifest hash mismatch: expected {expected_hash}, found {actual_hash}"
        )
    missing = [gamefile for gamefile in game_files if not Path(gamefile).exists()]
    if missing:
        raise FileNotFoundError(f"Manifest gamefile does not exist: {missing[0]}")
    return game_files


def collect_game_files(config: dict[str, Any], args: EvalArgs) -> list[str]:
    if args.gamefile_manifest:
        game_files = load_gamefile_manifest(args.gamefile_manifest)
        random.Random(args.seed).shuffle(game_files)
    else:
        collector = make_collector(config, args.split)
        game_files = list(collector.game_files)
    if args.task_type_filter:
        game_files = [
            gamefile
            for gamefile in game_files
            if get_task_type_from_gamefile(gamefile) == args.task_type_filter
        ]
    if args.limit is not None:
        game_files = game_files[: args.limit]
    return game_files


def split_evenly(items: list[str], chunks: int) -> list[list[str]]:
    chunks = max(1, min(chunks, len(items))) if items else 1
    result = [[] for _ in range(chunks)]
    for index, item in enumerate(items):
        result[index % chunks].append(item)
    return [chunk for chunk in result if chunk]


def summarize(rows: list[dict[str, Any]], args: EvalArgs) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["task_type"]].append(row)

    def total(subset: list[dict[str, Any]], key: str) -> int:
        return sum(int(row.get(key, 0) or 0) for row in subset)

    def mean(subset: list[dict[str, Any]], key: str) -> float:
        return statistics.mean(float(row.get(key, 0) or 0) for row in subset) if subset else 0.0

    def metric(subset: list[dict[str, Any]]) -> dict[str, Any]:
        episodes = len(subset)
        successful = [row for row in subset if row["success"]]
        successes = len(successful)
        errors = sum(1 for row in subset if row.get("error"))
        agent_turns = total(subset, "agent_turns")
        env_steps = total(subset, "env_steps")
        action_steps = total(subset, "action_steps")
        thought_steps = total(subset, "thought_steps")
        llm_calls = total(subset, "llm_calls")
        planning_calls = total(subset, "planning_calls")
        summary_calls = total(subset, "summary_calls")
        parse_failures = total(subset, "parse_failures")
        json_parse_failures = total(subset, "json_parse_failures")
        mode_violations = total(subset, "mode_violations")
        invalid_actions = total(subset, "invalid_actions")
        nothing_happens = total(subset, "nothing_happens")
        repeated_actions = total(subset, "repeated_actions")
        action_repairs = total(subset, "action_repairs")
        lookat_lamp_repairs = total(subset, "lookat_lamp_repairs")
        prompt_tokens = total(subset, "prompt_tokens")
        completion_tokens = total(subset, "completion_tokens")
        total_tokens = total(subset, "total_tokens")
        context_truncated_episodes = sum(
            any(step.get("context_truncated") for step in row.get("trajectory", []))
            for row in subset
        )
        context_budget_truncated_episodes = sum(
            any(step.get("context_budget_truncated") for step in row.get("trajectory", []))
            for row in subset
        )
        return {
            "episodes": episodes,
            "successes": successes,
            "errors": errors,
            "success_rate": successes / episodes if episodes else 0.0,
            "error_rate": errors / episodes if episodes else 0.0,
            "avg_steps": mean(subset, "env_steps"),
            "avg_env_steps": mean(subset, "env_steps"),
            "avg_success_env_steps": mean(successful, "env_steps"),
            "avg_agent_turns": mean(subset, "agent_turns"),
            "avg_action_steps": mean(subset, "action_steps"),
            "avg_thought_steps": mean(subset, "thought_steps"),
            "avg_llm_calls": mean(subset, "llm_calls"),
            "avg_prompt_tokens": mean(subset, "prompt_tokens"),
            "avg_completion_tokens": mean(subset, "completion_tokens"),
            "avg_total_tokens": mean(subset, "total_tokens"),
            "agent_turns": agent_turns,
            "env_steps": env_steps,
            "action_steps": action_steps,
            "thought_steps": thought_steps,
            "llm_calls": llm_calls,
            "planning_calls": planning_calls,
            "summary_calls": summary_calls,
            "parse_failures": parse_failures,
            "json_parse_failures": json_parse_failures,
            "mode_violations": mode_violations,
            "invalid_actions": invalid_actions,
            "nothing_happens": nothing_happens,
            "repeated_actions": repeated_actions,
            "action_repairs": action_repairs,
            "lookat_lamp_repairs": lookat_lamp_repairs,
            "repair_rate": action_repairs / action_steps if action_steps else 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "invalid_action_rate": invalid_actions / action_steps if action_steps else 0.0,
            "parse_failure_rate": parse_failures / agent_turns if agent_turns else 0.0,
            "repeated_action_rate": repeated_actions / action_steps if action_steps else 0.0,
            "context_truncated_episodes": context_truncated_episodes,
            "context_budget_truncated_episodes": context_budget_truncated_episodes,
            "avg_elapsed_sec": mean(subset, "elapsed_sec"),
        }

    by_task = {task: metric(groups.get(task, [])) for task in TASK_ORDER}
    overall = metric(rows)
    termination_counts = Counter(row.get("termination_reason", "unknown") for row in rows)
    prompt_bundles = load_prompts(args.prompt_file, args)
    payload = {
        "config": {
            "split": args.split,
            "model": args.model,
            "base_url": args.base_url,
            "prompt_file": args.prompt_file,
            "selected_prompt_keys": {
                task: list(prompt_bundles[task].keys) for task in TASK_ORDER
            },
            "agent_mode": args.agent_mode,
            "fewshot_strategy": args.fewshot_strategy,
            "examples_per_task": args.examples_per_task,
            "prompt_seed": args.prompt_seed,
            "include_admissible_actions": args.include_admissible_actions,
            "invalid_action_feedback": args.invalid_action_feedback,
            "output_format": args.output_format,
            "guided_decoding_backend": (
                args.guided_decoding_backend if args.output_format == "json" else None
            ),
            "history_mode": args.history_mode,
            "history_k": args.history_k,
            "summary_max_tokens": args.summary_max_tokens,
            "seed": args.seed,
            "gamefile_manifest": args.gamefile_manifest,
            "task_type_filter": args.task_type_filter,
            "lookat_lamp_action_repair": args.lookat_lamp_action_repair,
            "lookat_lamp_grammar_hint": args.lookat_lamp_grammar_hint,
            "gamefile_set_sha256": gamefile_set_sha256(
                [row["gamefile"] for row in rows]
            ),
            "max_steps": args.max_steps,
            "max_steps_definition": "maximum ALFWorld env.step calls; thought turns do not count",
            "max_agent_turns": args.max_agent_turns,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_context_chars": args.max_context_chars,
            "max_context_chars_definition": "character budget, not token budget",
            "max_consecutive_parse_failures": args.max_consecutive_parse_failures,
            "max_consecutive_parse_failures_definition": "safety cap for deterministic parse-failure loops; does not consume env_steps",
            "max_consecutive_thoughts": args.max_consecutive_thoughts,
            "max_consecutive_thoughts_definition": "safety cap for deterministic thought loops; does not consume env_steps",
            "workers": args.workers,
            "limit": args.limit,
            "placement_command_adapter": "normalize ReAct-style put OBJECT in/on RECEPTACLE to local ALFWorld move OBJECT to RECEPTACLE",
            "lookat_lamp_action_repair_definition": "optionally rewrite examine/look at TARGET with desklamp N to use desklamp N only after TARGET is held in look_at_obj_in_light tasks",
            "admissible_commands_in_prompt": args.include_admissible_actions,
        },
        "metric_definitions": {
            "agent_turns": "number of policy-decision LLM calls",
            "llm_calls": "decision, initial-planning, and history-summary LLM calls",
            "env_steps": "number of ALFWorld env.step calls",
            "thought_steps": "number of ReAct thought turns; no env.step call",
            "action_steps": "number of parsed action turns sent to env.step",
            "parse_failures": "outputs that cannot be normalized into the configured thought/action protocol",
            "invalid_actions": "parsed actions absent from current admissible_commands",
            "invalid_action_rate": "invalid_actions / action_steps",
            "parse_failure_rate": "parse_failures / agent_turns",
            "repeated_action_rate": "consecutive repeated actions / action_steps",
            "repair_rate": "action_repairs / action_steps",
            "avg_steps": "average env_steps",
            "token_metrics": "OpenAI-compatible API usage fields, including planner and summarizer calls",
        },
        "overall": overall,
        "by_task": by_task,
        "task_type_filter": args.task_type_filter,
        "action_repairs": overall["action_repairs"],
        "lookat_lamp_repairs": overall["lookat_lamp_repairs"],
        "repair_rate": overall["repair_rate"],
        "failure_count": sum(1 for row in rows if not row["success"]),
        "task_counts": dict(Counter(row["task_type"] for row in rows)),
        "termination_counts": dict(termination_counts),
    }
    return payload


METRICS_CSV_COLUMNS = [
    "task_type",
    "episodes",
    "successes",
    "success_rate",
    "avg_env_steps",
    "avg_success_env_steps",
    "avg_agent_turns",
    "avg_llm_calls",
    "avg_thought_steps",
    "invalid_action_rate",
    "parse_failure_rate",
    "repeated_action_rate",
    "repair_rate",
    "avg_prompt_tokens",
    "avg_completion_tokens",
    "avg_total_tokens",
    "invalid_actions",
    "parse_failures",
    "json_parse_failures",
    "mode_violations",
    "nothing_happens",
    "action_repairs",
    "lookat_lamp_repairs",
    "summary_calls",
    "context_truncated_episodes",
    "context_budget_truncated_episodes",
    "errors",
]


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: row["gamefile"])

    with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output_dir / "metrics_by_task.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
        writer.writeheader()

        def write_row(task_type: str, item: dict[str, Any]) -> None:
            row = {"task_type": task_type}
            for key in METRICS_CSV_COLUMNS[1:]:
                value = item[key]
                row[key] = f"{value:.6f}" if isinstance(value, float) else value
            writer.writerow(row)

        write_row("overall", summary["overall"])
        for task in TASK_ORDER:
            write_row(task, summary["by_task"][task])


def print_summary(summary: dict[str, Any]) -> None:
    print("\nEvaluation summary")
    overall = summary["overall"]
    print(
        f"overall: {overall['successes']}/{overall['episodes']} "
        f"success_rate={overall['success_rate']:.3f} "
        f"avg_env_steps={overall['avg_env_steps']:.2f} "
        f"avg_agent_turns={overall['avg_agent_turns']:.2f} "
        f"invalid_action_rate={overall['invalid_action_rate']:.3f} "
        f"parse_failure_rate={overall['parse_failure_rate']:.3f} "
        f"repairs={overall['action_repairs']} "
        f"errors={overall['errors']}"
    )
    for task in TASK_ORDER:
        item = summary["by_task"][task]
        print(
            f"{task}: {item['successes']}/{item['episodes']} "
            f"success_rate={item['success_rate']:.3f} "
            f"avg_env_steps={item['avg_env_steps']:.2f} "
            f"avg_agent_turns={item['avg_agent_turns']:.2f} "
            f"invalid_action_rate={item['invalid_action_rate']:.3f} "
            f"parse_failure_rate={item['parse_failure_rate']:.3f} "
            f"repairs={item['action_repairs']} "
            f"errors={item['errors']}"
        )


def main() -> None:
    args = to_eval_args(parse_args())
    ensure_data_dir()
    config = load_yaml(args.config)
    game_files = collect_game_files(config, args)
    if not game_files:
        raise RuntimeError(f"No games found for split={args.split}")

    print(
        f"Running split={args.split} games={len(game_files)} workers={args.workers} "
        f"max_env_steps={args.max_steps} max_agent_turns={args.max_agent_turns}"
    )
    chunks = split_evenly(game_files, args.workers)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    if len(chunks) == 1:
        rows = evaluate_slice(0, chunks[0], args)
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            future_to_meta = {
                executor.submit(evaluate_slice, worker_id, chunk, args): (worker_id, chunk)
                for worker_id, chunk in enumerate(chunks)
            }
            finished = 0
            for future in as_completed(future_to_meta):
                worker_id, chunk = future_to_meta[future]
                try:
                    part = future.result()
                except Exception as exc:
                    part = [
                        empty_episode(
                            worker_id * 1_000_000 + offset,
                            gamefile,
                            get_task_type_from_gamefile(gamefile),
                            repr(exc),
                            "worker_runtime_error",
                        )
                        for offset, gamefile in enumerate(chunk)
                    ]
                rows.extend(part)
                finished += len(part)
                print(f"finished {finished}/{len(game_files)} episodes", flush=True)

    elapsed = time.perf_counter() - started
    rows = sorted(rows, key=lambda row: row["gamefile"])
    for index, row in enumerate(rows):
        row["episode_index"] = index

    summary = summarize(rows, args)
    summary["wall_time_sec"] = elapsed
    output_dir = Path(args.output_dir)
    write_outputs(rows, summary, output_dir)
    print_summary(summary)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()
