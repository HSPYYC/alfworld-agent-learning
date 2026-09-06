from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import eval as evaluator
from scripts.run_task4 import load_config

ROOT = Path(__file__).resolve().parents[1]


def make_args(**overrides) -> evaluator.EvalArgs:
    values = {
        "config": "configs/base_config.yaml",
        "prompt_file": "prompts/alfworld_3prompts.json",
        "split": "eval_out_of_distribution",
        "output_dir": "results/test",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "qwen",
        "api_key": "EMPTY",
        "max_steps": 50,
        "max_agent_turns": 150,
        "max_tokens": 100,
        "temperature": 0.0,
        "max_context_chars": 24000,
        "max_consecutive_parse_failures": 8,
        "max_consecutive_thoughts": 24,
        "workers": 1,
        "limit": None,
        "timeout": 120.0,
        "trust_env": False,
        "examples_per_task": 2,
        "agent_mode": "react",
        "fewshot_strategy": "matched",
        "prompt_seed": 42,
        "include_admissible_actions": False,
        "invalid_action_feedback": False,
        "output_format": "free",
        "guided_decoding_backend": "xgrammar",
        "history_mode": "full",
        "history_k": 10,
        "summary_max_tokens": 160,
        "seed": 0,
        "gamefile_manifest": None,
    }
    values.update(overrides)
    return evaluator.EvalArgs(**values)


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompt_path = ROOT / "prompts/alfworld_3prompts.json"
        cls.raw = evaluator.load_prompt_source(cls.prompt_path)

    def test_matched_baseline_uses_official_order(self) -> None:
        args = make_args()
        bundles = evaluator.load_prompts(self.prompt_path, args)
        self.assertEqual(
            bundles["pick_and_place_simple"].keys,
            ("react_put_1", "react_put_0"),
        )
        prompt = evaluator.build_prompt(
            bundles["pick_and_place_simple"],
            "Your task is to: put an apple in cabinet.",
            args,
        )
        self.assertTrue(
            prompt.startswith(
                "Interact with a household to solve a task. Here are two examples.\n"
            )
        )

    def test_reasoning_modes_select_paired_example_families(self) -> None:
        react = evaluator.load_prompts(self.prompt_path, make_args())
        act = evaluator.load_prompts(
            self.prompt_path, make_args(agent_mode="act")
        )
        self.assertEqual(react["look_at_obj_in_light"].keys[0], "react_examine_1")
        self.assertEqual(act["look_at_obj_in_light"].keys[0], "act_examine_1")

    def test_random_nonmatching_is_deterministic_and_nonmatching(self) -> None:
        args = make_args(fewshot_strategy="random_nonmatching")
        first = evaluator.load_prompts(self.prompt_path, args)
        second = evaluator.load_prompts(self.prompt_path, args)
        for task, prefix in evaluator.PROMPT_PREFIXES.items():
            self.assertEqual(first[task].keys, second[task].keys)
            self.assertEqual(len(first[task].keys), 2)
            self.assertTrue(all(f"react_{prefix}_" not in key for key in first[task].keys))

    def test_mixed_and_zero_shot(self) -> None:
        mixed = evaluator.load_prompts(
            self.prompt_path, make_args(fewshot_strategy="mixed6")
        )
        zero = evaluator.load_prompts(
            self.prompt_path, make_args(fewshot_strategy="zero")
        )
        expected_keys = (
            "react_put_1",
            "react_clean_1",
            "react_heat_1",
            "react_cool_1",
            "react_examine_1",
            "react_puttwo_1",
        )
        self.assertTrue(all(bundle.keys == expected_keys for bundle in mixed.values()))
        self.assertTrue(all(not bundle.keys and not bundle.text for bundle in zero.values()))

    def test_mixed_target_last_only_changes_example_order(self) -> None:
        fixed = evaluator.load_prompts(
            self.prompt_path, make_args(fewshot_strategy="mixed6")
        )
        target_last = evaluator.load_prompts(
            self.prompt_path, make_args(fewshot_strategy="mixed6_target_last")
        )
        for task_type, prefix in evaluator.PROMPT_PREFIXES.items():
            self.assertCountEqual(target_last[task_type].keys, fixed[task_type].keys)
            self.assertEqual(target_last[task_type].keys[-1], f"react_{prefix}_1")

    def test_mixed_examples_normalize_to_local_grammar(self) -> None:
        local_action_patterns = [
            r"go to [a-z]+ \d+",
            r"(?:open|close|use) [a-z]+ \d+",
            r"take [a-z]+ \d+ from [a-z]+ \d+",
            r"move [a-z]+ \d+ to [a-z]+ \d+",
            r"put [a-z]+ \d+ into [a-z]+ \d+",
            r"put [a-z]+ \d+ in [a-z]+ \d+",
            r"(?:clean|heat|cool|slice) [a-z]+ \d+ with [a-z]+ \d+",
            r"examine [a-z]+ \d+",
            r"inventory",
            r"look",
        ]
        keys = (
            "react_put_1",
            "react_clean_1",
            "react_heat_1",
            "react_cool_1",
            "react_examine_1",
            "react_puttwo_1",
        )
        action_count = 0
        for key in keys:
            for line in self.raw[key].splitlines():
                if not line.startswith("> "):
                    continue
                parsed, kind, reason = evaluator.parse_model_output(line[2:], "free")
                self.assertNotEqual(kind, "parse_failure", msg=(key, line, reason))
                if kind == "thought":
                    continue
                action_count += 1
                self.assertTrue(
                    any(re.fullmatch(pattern, parsed) for pattern in local_action_patterns),
                    msg=(key, line, parsed),
                )
        self.assertEqual(action_count, 73)

    def test_json_examples_and_parser(self) -> None:
        transformed = evaluator.transform_example_to_json(self.raw["react_put_0"])
        turns = [line[2:] for line in transformed.splitlines() if line.startswith("> ")]
        self.assertTrue(turns)
        for turn in turns:
            self.assertEqual(set(json.loads(turn)), {"thought", "action"})

        thought = '{"thought":"check the fridge","action":""}'
        action = '{"thought":"","action":"put apple 1 in/on cabinet 1"}'
        self.assertEqual(
            evaluator.parse_model_output(thought, "json"),
            ("think: check the fridge", "thought", None),
        )
        self.assertEqual(
            evaluator.parse_model_output(action, "json"),
            ("move apple 1 to cabinet 1", "action", None),
        )
        self.assertEqual(
            evaluator.parse_model_output('{"thought":"","action":""}', "json")[2],
            "json_exclusive_fields",
        )


class LLMRequestTests(unittest.TestCase):
    def test_json_disables_newline_stop_only_for_guided_request(self) -> None:
        client = Mock()
        client.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(text='{"thought":"","action":"look"}')],
            usage=SimpleNamespace(
                prompt_tokens=3, completion_tokens=4, total_tokens=7
            ),
        )

        result = evaluator.llm(
            client, make_args(), "prompt", request_seed=7, guided_json=True
        )
        self.assertEqual(result.total_tokens, 7)
        self.assertIsNone(client.completions.create.call_args.kwargs["stop"])
        self.assertEqual(
            client.completions.create.call_args.kwargs["extra_body"][
                "guided_decoding_backend"
            ],
            "xgrammar",
        )

        evaluator.llm(
            client, make_args(), "prompt", request_seed=7, guided_json=False
        )
        self.assertEqual(client.completions.create.call_args.kwargs["stop"], ["\n"])
        self.assertIsNone(client.completions.create.call_args.kwargs["extra_body"])


class ContextAndManifestTests(unittest.TestCase):
    def test_full_and_recent_history(self) -> None:
        history = [f"turn-{index}" for index in range(5)]
        full, full_meta = evaluator.build_context("base", history, 1000)
        recent, recent_meta = evaluator.build_context(
            "base", history, 1000, history_mode="recent", history_k=2
        )
        self.assertEqual(full, "base\n" + "\n".join(history))
        self.assertFalse(full_meta["context_truncated"])
        self.assertEqual(recent, "base\nturn-3\nturn-4")
        self.assertEqual(recent_meta["history_items_policy_dropped"], 3)

    def test_summary_context_and_dynamic_suffix(self) -> None:
        history = ["old", "recent-a", "recent-b"]
        context, meta = evaluator.build_context(
            "base",
            history,
            1000,
            history_mode="summary",
            memory_summary="visited desk",
            summarized_items=1,
            dynamic_suffix="Current admissible actions: []",
        )
        self.assertIn("Memory summary: visited desk", context)
        self.assertNotIn("\nold\n", context)
        self.assertTrue(context.endswith("Current admissible actions: []"))
        self.assertEqual(meta["history_items_policy_dropped"], 1)

    def test_manifest_hash_and_count(self) -> None:
        gamefiles = evaluator.load_gamefile_manifest(
            ROOT / "configs/task4_eval_manifest.json"
        )
        self.assertEqual(len(gamefiles), 134)
        self.assertEqual(len(set(gamefiles)), 134)
        self.assertEqual(
            evaluator.gamefile_set_sha256(gamefiles),
            "9d7b36e410ca9fe4dc1fc5e20cb7750f823e271af775b4f650799e9da3740793",
        )

    def test_manifest_rejects_hash_mismatch(self) -> None:
        gamefile = json.loads(
            (ROOT / "results/task3_react_baseline/trajectories.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )["gamefile"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "gamefiles": [gamefile],
                        "expected_count": 1,
                        "gamefile_set_sha256": "bad",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                evaluator.load_gamefile_manifest(path)


class ExperimentConfigTests(unittest.TestCase):
    def test_matrix_contains_eleven_single_factor_configs(self) -> None:
        config = load_config(ROOT / "configs/task4_experiments.yaml")
        self.assertEqual(len(config["experiments"]), 11)
        self.assertEqual(config["experiments"]["baseline"]["overrides"], {})
        for name, item in config["experiments"].items():
            self.assertRegex(item["hypothesis"], r"[\u4e00-\u9fff]")
            if name != "baseline":
                self.assertEqual(len(item["overrides"]), 1)


    def test_frozen_random_prompt_map_matches_seeded_selection(self) -> None:
        config = load_config(ROOT / "configs/task4_experiments.yaml")
        frozen = config["experiments"]["random_nonmatching_2shot"]["fixed_prompt_keys"]
        bundles = evaluator.load_prompts(
            ROOT / "prompts/alfworld_3prompts.json",
            make_args(fewshot_strategy="random_nonmatching", prompt_seed=42),
        )
        self.assertEqual(
            {task: list(bundle.keys) for task, bundle in bundles.items()},
            frozen,
        )


if __name__ == "__main__":
    unittest.main()
