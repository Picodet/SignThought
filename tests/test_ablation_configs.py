import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
VARIANTS = {
    "full": {
        "seed42": "sign_phoenix_full.yaml",
        "mode": "dual",
        "thinking": True,
        "lambda_mono": 0.1,
        "lambda_cont": 0.2,
        "routing": True,
    },
    "no_thought": {
        "seed42": "sign_phoenix_no_thought.yaml",
        "mode": "visual_only",
        "thinking": False,
        "lambda_mono": 0.0,
        "lambda_cont": 0.0,
        "routing": False,
    },
    "no_thought_dual_evidence": {
        "seed42": "sign_phoenix_no_thought_dual_evidence.yaml",
        "mode": "dual_evidence",
        "thinking": False,
        "lambda_mono": 0.0,
        "lambda_cont": 0.0,
        "routing": False,
    },
}


def config_path(variant, seed):
    if seed == 42:
        filename = VARIANTS[variant]["seed42"]
    else:
        filename = "sign_phoenix_{}_seed{}.yaml".format(variant, seed)
    return ROOT / "configs" / "ablation" / filename


class AblationConfigTests(unittest.TestCase):
    def load(self, variant, seed):
        path = config_path(variant, seed)
        self.assertTrue(path.is_file(), "Missing config: {}".format(path))
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_all_variants_use_stable_baseline_v1(self):
        loaded = {
            (variant, seed): self.load(variant, seed)
            for variant in VARIANTS
            for seed in SEEDS
        }
        reference = loaded[("full", 42)]

        for (variant, seed), cfg in loaded.items():
            with self.subTest(variant=variant, seed=seed):
                self.assertEqual(cfg["training"]["random_seed"], seed)
                self.assertEqual(cfg["data"], reference["data"])
                self.assertEqual(cfg["testing"]["translation_beam_sizes"], [7])
                self.assertEqual(cfg["testing"]["translation_beam_alphas"], [4])

                training = cfg["training"]
                expected_training = {
                    "translation_loss_weight": 1.0,
                    "eval_metric": "bleu",
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "num_valid_log": 6,
                    "epochs": 500,
                    "early_stopping_metric": "eval_metric",
                    "batch_type": "sentence",
                    "translation_normalization": "batch",
                    "eval_translation_beam_size": 7,
                    "eval_translation_beam_alpha": 4,
                    "decrease_factor": 0.8,
                    "patience": 1000,
                    "label_smoothing": 0.1,
                    "warmup_step": 2000,
                    "mono_margin": 1.0,
                }
                for key, value in expected_training.items():
                    self.assertEqual(training[key], value, "{} mismatch".format(key))

                # The stable baseline intentionally relies on main/helpers.py
                # to inject these effective runtime values.
                self.assertNotIn("betas", training)
                self.assertNotIn("weight_decay", training)
                self.assertIn("_s{}_auto".format(seed), training["model_dir"])
                self.assertIn("_s{}".format(seed), training["wandb"]["run_name"])

                model = cfg["model"]
                for component in ("encoder", "decoder"):
                    part = model[component]
                    self.assertEqual(part["num_layers"], 1)
                    self.assertEqual(part["num_heads"], 8)
                    self.assertEqual(part["hidden_size"], 256)
                    self.assertEqual(part["ff_size"], 1024)
                    self.assertEqual(part["dropout"], 0.1)
                    self.assertEqual(part["embeddings"]["embedding_dim"], 256)
                    self.assertFalse(part["embeddings"]["scale"])
                    self.assertEqual(part["embeddings"]["dropout"], 0.1)
                    self.assertEqual(part["embeddings"]["norm_type"], "batch")
                    self.assertEqual(part["embeddings"]["activation_type"], "softsign")

                thinking = cfg["thinking"]
                self.assertEqual(thinking["K"], 8)
                self.assertEqual(thinking["num_segments"], 8)
                self.assertEqual(thinking["num_layers"], 2)
                self.assertEqual(thinking["num_heads"], 8)
                self.assertEqual(thinking["hidden_size"], 256)
                self.assertEqual(thinking["ff_size"], 1024)
                self.assertEqual(thinking["dropout"], 0.1)
                self.assertTrue(thinking["causal_self_attn"])
                self.assertEqual(thinking["sinkhorn_iters"], 5)
                self.assertEqual(thinking["sinkhorn_tau"], 1.0)
                self.assertEqual(thinking["segmentation_temperature"], 0.08)
                self.assertEqual(thinking["retrieval_mode"], "dense")
                self.assertEqual(thinking["sparse_topk"], 4)
                self.assertEqual(thinking["deformable_radius"], 2)

                expected = VARIANTS[variant]
                self.assertEqual(model["decoder"]["cross_attention_mode"], expected["mode"])
                self.assertEqual(thinking["enabled"], expected["thinking"])
                self.assertEqual(training["lambda_mono"], expected["lambda_mono"])
                self.assertEqual(training["lambda_cont"], expected["lambda_cont"])
                self.assertEqual(thinking["enable_decoder_routing"], expected["routing"])

    def test_runtime_overrides_match_stable_baseline(self):
        helpers = (ROOT / "main" / "helpers.py").read_text(encoding="utf-8")
        builders = (ROOT / "main" / "builders.py").read_text(encoding="utf-8")
        self.assertIn('"betas": [0.95, 0.998]', helpers)
        self.assertIn('"scheduling": "plateau"', helpers)
        self.assertIn('"weight_decay": 0.003', helpers)
        self.assertIn("return torch.optim.Adam(", builders)


if __name__ == "__main__":
    unittest.main()
