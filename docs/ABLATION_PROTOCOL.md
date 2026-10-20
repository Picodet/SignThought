# SignThought stable-baseline-based ablation protocol

## Source of truth
Use the user-supplied **SignThought Stable Baseline V1.0** config package and `基准 YAML.yaml` as the training-hyperparameter source. Do not substitute the paper-only 200-epoch / 2-layer setup for these ablation runs.

The uploaded baseline package identifies upstream source `fletcherjiang/SignThought` at commit `f9ce7d175308e8c736df53dd29493503239738ea`. This working repository is `Picodet/SignThought`; its original `main` remains at `fdce86a9b13f9cd82ac1719a73c6b0244d171cb4`. The ablation branches preserve the local ablation implementation rather than resetting to the upstream pin. We checked that the current branch `main/helpers.py` and `main/builders.py` have the runtime optimizer/scheduler behavior documented in the supplied package, but the different repository/commit pins mean this is a matched parameter recipe, not a claim that every source line is identical.

## Stable baseline parameters (shared across conditions)
- PHOENIX2014T official train/dev/test splits; 1024-dimensional pre-extracted features as expected by the source.
- Seeds: 42, 123, 2026; use paired seeds across all three ablation conditions.
- 500 epochs; Adam learning rate 0.001; batch size 32; validation logs 6; patience 1000; decrease factor 0.8.
- Translation loss weight 1.0; BLEU selection metric; sentence batching; batch translation normalization; label smoothing 0.1.
- Beam size 7 and beam alpha 4 for validation/test decoding.
- Encoder and decoder each have 1 Transformer layer, 8 heads, hidden size 256, FF size 1024, dropout 0.1. Embeddings use dimension 256, scale=false, dropout 0.1, batch norm and softsign.
- Thinking settings: K=8, M=8 segments, 2 Thinking layers, 8 heads, hidden size 256, FF size 1024, dropout 0.1, causal self-attention, Sinkhorn iterations 5, tau 1.0, segmentation temperature 0.08, dense retrieval, sparse_topk=4, deformable_radius=2, routing bias scale 1.0.
- Preserve `warmup_step: 2000` for configuration fidelity, but do not claim warmup is active: the runtime scheduler is plateau.

## Runtime-effective optimizer details (important)
The YAML uses `optimizer: adamw` because that is what the stable baseline uses, but the current source implementation resolves this key to `torch.optim.Adam`. `main/helpers.py` injects `betas=[0.95, 0.998]`, `scheduling="plateau"`, and `weight_decay=0.003` into training config. These effective values intentionally are not duplicated as YAML keys because the helper overwrites them. Do not silently switch to AdamW, betas (0.9, 0.998), or a warmup scheduler within this architecture ablation.

The supplied baseline package reports best Dev BLEU-4 >=25 for seeds 42/123/2026 and Test BLEU-4 of 27.65, 26.81, and 26.82 (mean 27.09). These are reference baseline results from the supplied package, not results of the ablation runs and not a guarantee for this different repository pin.

## Ablation variants
- `full`: Thinking enabled; `cross_attention_mode=dual`; decoder routing enabled; `lambda_mono=0.1`, `lambda_cont=0.2`.
- `no_thought`: Thinking not instantiated; direct visual-only decoder; routing disabled; structural losses set to zero because the corresponding structures do not exist.
- `no_thought_dual_evidence`: Thinking not instantiated; keeps two decoder evidence-attention stages by replacing Thought retrieval with a second visual-evidence retrieval; routing disabled; structural losses zero. This is an explicit implementation control inspired by the paper's no-Thought dual-stream ablation, not a claim that the paper specifies this exact replacement.

## Config matrix (available on every ablation branch)
Seed 42 uses the original concise filenames:
- `configs/ablation/sign_phoenix_full.yaml`
- `configs/ablation/sign_phoenix_no_thought.yaml`
- `configs/ablation/sign_phoenix_no_thought_dual_evidence.yaml`

Seeds 123 and 2026 use:
- `configs/ablation/sign_phoenix_full_seed123.yaml`, `configs/ablation/sign_phoenix_full_seed2026.yaml`
- `configs/ablation/sign_phoenix_no_thought_seed123.yaml`, `configs/ablation/sign_phoenix_no_thought_seed2026.yaml`
- `configs/ablation/sign_phoenix_no_thought_dual_evidence_seed123.yaml`, `configs/ablation/sign_phoenix_no_thought_dual_evidence_seed2026.yaml`

Each variant/seed has a distinct `model_dir` containing `auto`, which the source replaces with a timestamp, and a distinct W&B run name. W&B is enabled to match the supplied baseline; if unavailable, disable it consistently and record that environment change in the experiment manifest.

## Run examples
```bash
CUDA_VISIBLE_DEVICES=0 python -m main train configs/ablation/sign_phoenix_full.yaml --gpu_id 0
CUDA_VISIBLE_DEVICES=0 python -m main train configs/ablation/sign_phoenix_no_thought.yaml --gpu_id 0
CUDA_VISIBLE_DEVICES=0 python -m main train configs/ablation/sign_phoenix_no_thought_dual_evidence.yaml --gpu_id 0
```
Use the corresponding `_seed123.yaml` and `_seed2026.yaml` files for the other paired seeds. Select checkpoints on Dev BLEU only and evaluate Test after checkpoint selection. Report per-seed results and mean/std, BLEU-4, ROUGE, parameter count, peak memory and decoding latency.

## Validation and limits
GitHub Actions checks Python compilation and validates all nine config combinations and the fixed runtime optimizer/scheduler settings. This does not substitute for GPU one-batch forward/backward, decoding smoke tests, or full training. Before long runs, verify that No-Thought models have `model.thinking is None`, no Thought attention parameters, no routing prior, and zero structural-loss weights.

## Main branch
No changes have been committed to `main`; its SHA remains `fdce86a9b13f9cd82ac1719a73c6b0244d171cb4`. GitHub API access cannot enable branch protection for this private repository (403 / plan or permission restriction); a repository admin must enable PR-required branch protection in GitHub settings.
