from __future__ import annotations

import unittest
from pathlib import Path

from scripts import run_reference_baseline_probe as probe


ROOT = Path(__file__).resolve().parents[1]


class ReferenceBaselineProbeTests(unittest.TestCase):
    def test_selected_keys(self) -> None:
        self.assertEqual(
            probe.selected_keys("matched", "pick_and_place_simple"),
            ("react_put_0", "react_put_1"),
        )
        expected_mixed = (
            "react_put_0",
            "react_clean_0",
            "react_heat_0",
            "react_cool_0",
            "react_examine_0",
            "react_puttwo_0",
        )
        for task_type in probe.core.TASK_ORDER:
            self.assertEqual(probe.selected_keys("mixed6", task_type), expected_mixed)

    def test_examples_are_reconstructed_as_chat_turns(self) -> None:
        messages, _ = probe.load_demo_messages(
            ROOT / "prompts/mpo_reference_alfworld_icl.json", "matched"
        )
        selected = messages["pick_and_place_simple"]
        self.assertEqual(selected[0]["role"], "system")
        self.assertFalse(any("> think:" in item["content"] for item in selected))
        assistants = [item["content"] for item in selected if item["role"] == "assistant"]
        self.assertTrue(assistants)
        self.assertTrue(all(content.count("Action:") == 1 for content in assistants))

    def test_action_parser_requires_one_action_and_normalizes_placement(self) -> None:
        self.assertEqual(
            probe.parse_reference_action(
                "Thought: The apple is held.\nAction: put apple 1 in/on desk 1"
            ),
            ("move apple 1 to desk 1", None),
        )
        self.assertEqual(
            probe.parse_reference_action("Thought: I should search.")[1],
            "missing_action_line",
        )
        self.assertEqual(
            probe.parse_reference_action("Action: look\nAction: inventory")[1],
            "multiple_action_lines",
        )

    def test_probe_manifest_is_balanced_and_fixed(self) -> None:
        gamefiles = probe.load_probe_gamefiles(
            ROOT / "configs/reference_baseline_probe_manifest.json"
        )
        counts = {
            task: sum(
                probe.core.get_task_type_from_gamefile(path) == task for path in gamefiles
            )
            for task in probe.core.TASK_ORDER
        }
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual(
            probe.core.gamefile_set_sha256(gamefiles),
            "4c7a1f891262be9b5e13de4abb18d0f8c56caf5f2fea8879885b09c4a48cb1f5",
        )


if __name__ == "__main__":
    unittest.main()
