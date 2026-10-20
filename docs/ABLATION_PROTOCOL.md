# SignThought ablation protocol (paper-aligned)

## Immutable baseline and branch scope
- Original baseline commit: `fdce86a9b13f9cd82ac1719a73c6b0244d171cb4`.
- Shared implementation commit: `0d3490d302d93269d4c9c3e544180027517a3a0d`.
- The original `main` branch is not modified by this work. All writes are confined to `ablation/*` branches.

## What the paper actually specifies
The supplied SignThought paper, Appendix A.1 and Table 4, reports 2-layer encoder and decoder, 2 Thinking layers, K=8 Thought slots, M=16 segment tokens, 200 epochs, d=256, Adam lr=1e-3 with betas (0.9, 0.998), weight decay 3e-3, batch size 32, label smoothing 0.1, dropout 0.1, and 2000 warmup steps. The experiment configs now follow these settings. The repository's original YAML differed (1 encoder/decoder layer, M=8, 500 epochs, and no paper weight decay/betas), so these are paper-protocol runs rather than reproductions of the repository's old default run.

Table 4 ID 1 turns off latent Thinking, causal Thought updates, routing, prior injection, and both structural losses while keeping the dual-stream decoder. The paper does not specify in enough detail what the first decoder stream consumes after Thinking is removed. Therefore the exact replacement is an implementation inference, not a claim about the authors' unique code.

## Branches and variants
- `ablation/full-v1`: full model; Thought branch and structural regularizers enabled.
- `ablation/no-thought-dual-evidence-v1`: primary Table-4-ID-1-inspired variant. It removes all Thought computation and routing, but retains a two-cross-attention decoder by replacing the first stream with visual-evidence attention over encoder memory. This replacement is explicit and reproducible, but remains an inference because the paper does not specify that exact replacement.
- `ablation/no-thought-v1`: single-stream/direct-visual decoder; closer to the paper's ID 4 decoder simplification and an additional control, not the primary ID 1 reproduction.
- `ablation/common-v1`: shared implementation, all three configs, protocol documentation, tests, and smoke workflow.

Configs:
- `configs/ablation/sign_phoenix_full.yaml`
- `configs/ablation/sign_phoenix_no_thought_dual_evidence.yaml`
- `configs/ablation/sign_phoenix_no_thought.yaml`

## Fair-comparison protocol
Use identical dataset splits, vocabulary/data preprocessing, feature files, model width, optimizer, learning-rate schedule, batch size, decoding configuration, and epoch budget. Full uses lambda_mono=0.1 and lambda_cont=0.2 as specified in the paper. No-Thought variants set both to zero because the associated Thought structures do not exist. Run seeds 42, 123, and 2026 with distinct output directories. Select checkpoints on dev only; use the test set only after selection. Report per-seed results, mean/std, BLEU-4, ROUGE, parameter count, peak memory, and decoding latency.

## Checks before long runs
1. The GitHub Actions smoke workflow runs `compileall` and validates YAML configuration consistency.
2. Run a local one-batch forward/backward and dev decoding test before full runs.
3. In No-Thought variants assert `model.thinking is None`, no `thought_trg_att` parameters exist, routing is disabled, and structural loss weights are zero.
4. Verify the actual optimizer class. In this repository, the `adamw` config key currently maps to `torch.optim.Adam`, not `torch.optim.AdamW`. This pre-existing mismatch is documented rather than silently changed inside a structural ablation; resolve it identically for every condition in a separate optimizer-correction experiment if strict AdamW is required.
5. Save the exact config, branch/commit SHA, seed, best checkpoint, and test output for every run.

## Main branch protection
The GitHub integration cannot configure branch-protection rules for this private repository (the protection endpoint returns 403, and rulesets report a plan/permission restriction). This work has not written to `main`. Repository-level enforcement still needs an administrator to enable protection in GitHub settings.
