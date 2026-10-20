import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "sign_phoenix_full.yaml": ("dual", True, 0.1, 0.2),
    "sign_phoenix_no_thought.yaml": ("visual_only", False, 0.0, 0.0),
    "sign_phoenix_no_thought_dual_evidence.yaml": ("dual_evidence", False, 0.0, 0.0),
}


class AblationConfigTests(unittest.TestCase):
    def test_configs_share_training_and_model_dimensions(self):
        configs = {}
        for filename in CASES:
            path = ROOT / "configs" / "ablation" / filename
            with path.open("r", encoding="utf-8") as handle:
                configs[filename] = yaml.safe_load(handle)

        full = configs["sign_phoenix_full.yaml"]
        for cfg in configs.values():
            self.assertEqual(cfg["data"]["train"], full["data"]["train"])
            self.assertEqual(cfg["data"]["dev"], full["data"]["dev"])
            self.assertEqual(cfg["data"]["test"], full["data"]["test"])
            self.assertEqual(cfg["model"]["encoder"], full["model"]["encoder"])
            self.assertEqual(cfg["model"]["decoder"]["num_layers"], full["model"]["decoder"]["num_layers"])
            self.assertEqual(cfg["model"]["decoder"]["num_heads"], full["model"]["decoder"]["num_heads"])
            self.assertEqual(cfg["training"]["epochs"], 200)
            self.assertEqual(cfg["training"]["optimizer"], full["training"]["optimizer"])
            self.assertEqual(cfg["training"]["learning_rate"], full["training"]["learning_rate"])

    def test_thought_modes_and_losses_are_explicit(self):
        for filename, (mode, enabled, lambda_mono, lambda_cont) in CASES.items():
            with (ROOT / "configs" / "ablation" / filename).open("r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle)
            self.assertEqual(cfg["model"]["decoder"]["cross_attention_mode"], mode)
            self.assertEqual(cfg["thinking"]["enabled"], enabled)
            self.assertEqual(cfg["training"]["lambda_mono"], lambda_mono)
            self.assertEqual(cfg["training"]["lambda_cont"], lambda_cont)
            if not enabled:
                self.assertFalse(cfg["thinking"]["enable_decoder_routing"])


if __name__ == "__main__":
    unittest.main()
