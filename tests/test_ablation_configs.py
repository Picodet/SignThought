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
    def load(self, filename):
        with (ROOT / "configs" / "ablation" / filename).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_configs_share_paper_training_protocol(self):
        configs = {name: self.load(name) for name in CASES}
        full = configs["sign_phoenix_full.yaml"]
        for cfg in configs.values():
            self.assertEqual(cfg["data"]["train"], full["data"]["train"])
            self.assertEqual(cfg["data"]["dev"], full["data"]["dev"])
            self.assertEqual(cfg["data"]["test"], full["data"]["test"])
            self.assertEqual(cfg["model"]["encoder"]["num_layers"], 2)
            self.assertEqual(cfg["model"]["decoder"]["num_layers"], 2)
            self.assertEqual(cfg["model"]["encoder"]["hidden_size"], 256)
            self.assertEqual(cfg["model"]["decoder"]["hidden_size"], 256)
            self.assertEqual(cfg["training"]["epochs"], 200)
            self.assertEqual(cfg["training"]["batch_size"], 32)
            self.assertEqual(cfg["training"]["optimizer"], full["training"]["optimizer"])
            self.assertEqual(cfg["training"]["learning_rate"], 0.001)
            self.assertEqual(cfg["training"]["betas"], [0.9, 0.998])
            self.assertEqual(cfg["training"]["weight_decay"], 0.003)
            self.assertEqual(cfg["training"]["label_smoothing"], 0.1)
            self.assertEqual(cfg["training"]["warmup_step"], 2000)
            self.assertEqual(cfg["model"]["decoder"]["num_heads"], 8)
            self.assertEqual(cfg["thinking"]["K"], 8)
            self.assertEqual(cfg["thinking"]["num_layers"], 2)
            self.assertEqual(cfg["thinking"]["num_segments"], 16)

    def test_thought_modes_and_losses_are_explicit(self):
        for filename, (mode, enabled, lambda_mono, lambda_cont) in CASES.items():
            cfg = self.load(filename)
            self.assertEqual(cfg["model"]["decoder"]["cross_attention_mode"], mode)
            self.assertEqual(cfg["thinking"]["enabled"], enabled)
            self.assertEqual(cfg["training"]["lambda_mono"], lambda_mono)
            self.assertEqual(cfg["training"]["lambda_cont"], lambda_cont)
            if not enabled:
                self.assertFalse(cfg["thinking"]["enable_decoder_routing"])


if __name__ == "__main__":
    unittest.main()
