# SignThought ablation protocol (source-aligned)

## Immutable baseline
- Baseline source commit: `fdce86a9b13f9cd82ac1719a73c6b0244d171cb4`.
- Shared implementation commit: `0d3490d302d93269d4c9c3e544180027517a3a0d`.
- `main` is not modified by this work. All ablation changes are confined to `ablation/*` branches.

## Paper alignment and limits
The paper's ablation table describes disabling Thinking while retaining a dual-stream decoder. It does not specify what replaces the Thought stream after removal. Therefore this implementation reports two explicit, reproducible No-Thought variants rather than claiming either is the unique author implementation:
1. `visual_only`: decoder self-attention -> direct visual cross-attention -> FFN.
2. `dual_evidence`: decoder self-attention -> visual cross-attention -> a second visual cross-attention -> FFN. This is a capacity/control baseline, not a literal reconstruction of an unspecified paper detail.
3. `dual`: original Thought cross-attention plus visual cross-attention.

## Configs
- `configs/ablation/sign_phoenix_full.yaml`
- `configs/ablation/sign_phoenix_no_thought.yaml`
- `configs/ablation/sign_phoenix_no_thought_dual_evidence.yaml`

Configs use the same data splits, vocabulary creation, encoder/decoder width, dropout, optimizer configuration, scheduler settings, decoding parameters, and 200-epoch budget. 200 epochs follows the paper's reported training protocol; the uploaded project default was 500 epochs, so these are paper-protocol configs rather than a claim that they reproduce the repository's prior default run. Run all three with seeds 42, 123, and 2026 by overriding `training.random_seed` per run and using a distinct output directory.

## Interpretation
- Full vs Visual-only estimates the overall benefit of the Thought pathway, including the extra cross-attention stream and its learned representation.
- Visual-only vs Dual-evidence estimates the effect of an extra visual retrieval block / additional capacity.
- Full vs Dual-evidence compares Thought-conditioned retrieval against a matched second visual retrieval. Parameter counts are not perfectly matched because Thought has its own parameters; report actual parameter counts and latency.
- Select checkpoints on dev only. Evaluate test once after selection. Report each seed, mean and standard deviation, BLEU/CHRF/ROUGE, parameter count, peak memory, and decoding latency.

## Validation before long runs
1. Confirm all three configs parse and instantiate.
2. In No-Thought runs assert `model.thinking is None`, no `thought_trg_att` parameters exist, and all `lambda_mono/lambda_cont` are zero.
3. Confirm direct and beam search pass no Thought tensors.
4. Run one-batch forward/backward and one dev decode for each variant.
5. Compare parameter names and actual optimizer type. In this repository `optimizer: adamw` currently maps to `torch.optim.Adam` (not `torch.optim.AdamW`); this existing behavior is intentionally not silently changed in an architecture ablation. Fix it only in a separately named optimizer-correction experiment, or correct it identically across every condition before the final study.
6. Keep original seed, config, commit SHA, best checkpoint, and test output with every run.

## Platform branch protection
The GitHub integration cannot read or configure GitHub branch-protection rules in this repository (API returns 403). This work protects `main` operationally by making commits only to the explicitly named `ablation/*` branches and never updating `main`; repository-level protection still needs to be enabled by a repository admin in GitHub settings.
