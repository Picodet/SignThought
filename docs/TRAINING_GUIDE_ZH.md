# SignThought 消融实验训练手册（稳定基准 V1.0）

> 适用仓库：`Picodet/SignThought`  
> 适用分支：`ablation/full-v1`、`ablation/no-thought-v1`、`ablation/no-thought-dual-evidence-v1`  
> 公共集成分支：`ablation/common-v1`  
> 数据集：PHOENIX2014T，官方 train/dev/test 划分，1024 维预提取视觉特征  
> 文档目的：在固定训练配方与数据划分的前提下，比较 Full、No-Thought 单视觉流和 No-Thought 双证据流。

## 1. 实验原则

本实验以用户提供的《SignThought 稳定训练基准 V1.0》完整配置包及 `基准 YAML.yaml` 为超参数依据。不要把此前按论文表格整理的 200 Epoch、Encoder/Decoder 各 2 层的配置混入本轮稳定基准实验。

三个实验条件必须使用相同的数据划分、训练配方、随机种子集合和评估设置。仅允许改变与消融定义直接相关的 Thought、Decoder Cross-Attention 模式、Routing 和对应结构损失。不要为了让某个消融组更容易收敛而单独调整学习率、训练轮数或解码参数。

### 实验条件

| 条件 | Thought | Decoder Cross-Attention | Routing | `lambda_mono` | `lambda_cont` |
|---|---|---|---|---:|---:|
| Full | 启用 | `dual` | 启用 | 0.1 | 0.2 |
| No-Thought 单视觉流 | 关闭/不实例化 | `visual_only` | 关闭 | 0.0 | 0.0 |
| No-Thought 双证据流 | 关闭/不实例化 | `dual_evidence` | 关闭 | 0.0 | 0.0 |

No-Thought 双证据流通过第二次视觉证据检索保留两阶段证据注意力。这是当前仓库定义的对照实现，不应描述为论文唯一指定的实现细节。消融报告应分别呈现三个条件，不能把公共集成分支当作第四种模型条件。

## 2. 固定训练配置

所有条件和所有随机种子都遵循以下配置：

| 参数 | 固定值 |
|---|---|
| 数据集 | PHOENIX2014T |
| 特征维度 | 1024 |
| 随机种子 | 42、123、2026 |
| 最大训练轮数 | 500 Epoch |
| 初始学习率 | 0.001 |
| Batch size | 32 |
| 验证日志配置 | `num_valid_log: 6` |
| 早停监控指标 | `eval_metric`（BLEU） |
| Patience | 1000 |
| 学习率衰减系数 | 0.8 |
| Translation loss weight | 1.0 |
| Label smoothing | 0.1 |
| 验证/测试 beam size | 7 |
| 验证/测试 beam alpha | 4 |
| Encoder / Decoder 层数 | 各 1 层 |
| Attention heads | 8 |
| Hidden size / FF size | 256 / 1024 |
| Dropout | 0.1 |
| Thought slots (K) | 8 |
| Thinking layers | 2 |
| Segments (M) | 8 |
| Segmentation temperature | 0.08 |
| Sinkhorn iterations / tau | 5 / 1.0 |
| Retrieval mode | dense |
| Sparse top-k | 4 |
| Deformable radius | 2 |

Embedding 配置同样固定：维度 256、scale=false、dropout 0.1、BatchNorm 和 softsign。

### 2.1 必须核对的真实运行参数

不能只根据 YAML 中的名称判断实际训练行为。当前分支源码中：

- YAML 的 `optimizer: adamw` 在 `main/builders.py` 中实际映射到 `torch.optim.Adam`，不是 `torch.optim.AdamW`。
- `main/helpers.py` 的固定训练配置会覆盖 `betas=[0.95, 0.998]`、`scheduling="plateau"` 和 `weight_decay=0.003`。
- 配置中的 `warmup_step: 2000` 会保留，但在 `plateau` 调度器下不代表执行 Warmup。

本轮架构消融应保留这些实际行为，不要自行将优化器切换为 AdamW、改成其他 betas 或切换调度器。如果今后需要比较这些因素，应另建优化器/调度器消融，而不是混入 Thought 消融。

## 3. 代码版本与分支选择

稳定基准配置包记录的上游源码锚点是 `fletcherjiang/SignThought` 的 commit：

```text
f9ce7d175308e8c736df53dd29493503239738ea
```

本实验操作仓库为 `Picodet/SignThought`。消融分支保留该仓库中的消融实现，并按稳定基准对齐训练配方；这不等于整个仓库源码与上游锚点逐行相同。正式记录结果时必须保存当前分支名和 `git rev-parse HEAD` 的输出。

建议使用三个实验分支各自的对应配置：

- `ablation/full-v1`：运行 Full 配置。
- `ablation/no-thought-v1`：运行 No-Thought 单视觉流配置。
- `ablation/no-thought-dual-evidence-v1`：运行 No-Thought 双证据流配置。
- `ablation/common-v1`：公共实现和集成参考分支，不作为第四种实验条件。

所有分支均包含完整的 9 份配置矩阵，便于核对，但正式统计时只使用与所选分支相匹配的配置，避免重复计数。

## 4. 环境准备

稳定基准包记录的环境为 Python 3.9.25、PyTorch 2.4.0+cu121、torchtext 0.6.0，验证硬件为 RTX A6000。它们是基准记录中的环境信息，不保证其他环境天然得到相同结果。

先按仓库 README 和项目使用的环境说明安装依赖。启动前确认当前分支、Python 和 PyTorch：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

如果 CUDA 不可用，先解决环境问题，不要直接启动正式训练。

## 5. 数据准备

配置使用相对路径 `data_path: ./data/`，从仓库根目录启动训练。预期目录为：

```text
data/
└── PHOENIX2014T/
    ├── train/
    ├── dev/
    └── test/
```

目录中应放置项目要求的 PHOENIX2014T 1024 维预提取特征文件（官方数据划分）。数据应来自同一特征包，不能在不同实验条件间混用不同特征版本或划分。

启动前至少确认目录存在、train/dev/test 均非空，并检查特征文件能被项目数据加载器读取。不要将原始视频直接当作该配置所需的预提取特征文件使用。

## 6. 训练命令

请从仓库根目录执行命令。CLI 默认 GPU ID 为 3，因此应显式传入 `--gpu_id`。以下示例使用 GPU 0；若使用其他 GPU，将所有命令中的编号统一替换为目标编号。

### 6.1 Full：三个随机种子

先切换到 Full 分支：

```bash
git switch ablation/full-v1
```

然后依次运行：

```bash
python -m main train configs/ablation/sign_phoenix_full.yaml --gpu_id 0
python -m main train configs/ablation/sign_phoenix_full_seed123.yaml --gpu_id 0
python -m main train configs/ablation/sign_phoenix_full_seed2026.yaml --gpu_id 0
```

### 6.2 No-Thought 单视觉流：三个随机种子

```bash
git switch ablation/no-thought-v1
python -m main train configs/ablation/sign_phoenix_no_thought.yaml --gpu_id 0
python -m main train configs/ablation/sign_phoenix_no_thought_seed123.yaml --gpu_id 0
python -m main train configs/ablation/sign_phoenix_no_thought_seed2026.yaml --gpu_id 0
```

### 6.3 No-Thought 双证据流：三个随机种子

```bash
git switch ablation/no-thought-dual-evidence-v1
python -m main train configs/ablation/sign_phoenix_no_thought_dual_evidence.yaml --gpu_id 0
python -m main train configs/ablation/sign_phoenix_no_thought_dual_evidence_seed123.yaml --gpu_id 0
python -m main train configs/ablation/sign_phoenix_no_thought_dual_evidence_seed2026.yaml --gpu_id 0
```

建议同一时间只在一张 GPU 上运行一个训练任务，按上述命令顺序完成 9 次训练。这样能减少显存争用和并发造成的性能波动。若必须并行，需记录每个任务的 GPU 型号、设备编号和并发情况。

## 7. 输出、Checkpoint 与评估

每个配置的 `training.model_dir` 都是独立目录，并带有 `auto` 标记；加载配置时，代码会将其替换为时间戳。训练日志、配置副本、词表、Checkpoint 和预测结果会写入该目录。不要手工把多个运行的结果合并到同一输出目录。

训练管理器会按验证指标更新最佳 Checkpoint。训练结束后，当前训练入口会使用最佳验证 Checkpoint 自动运行评估，并将输出写入类似 `best.IT_XXXXXXXX` 的目录。

正式结果遵循以下原则：

1. 以开发集 BLEU 指标选择最佳 Checkpoint。
2. 测试集指标只用于最终报告，不得用测试集结果选择模型或调参。
3. 不要将不同随机种子或不同消融条件的 Checkpoint 混用。
4. 若需要单独执行测试，必须传入对应运行的 Checkpoint，并使用该实验条件的同一份 YAML 配置；由于 `auto` 会在每次加载配置时重新展开，单独测试时应显式指定 Checkpoint 路径。

W&B 配置默认启用。若使用 W&B，请先完成认证并确认项目可写。如果实验环境不使用 W&B，可将对应配置中的 `wandb.enabled` 统一关闭，但必须对全部实验一致处理，并在实验记录中注明。当前训练代码在 W&B 包不可用时会跳过集成；包已安装但未完成认证时，应先解决认证问题。

## 8. 运行前检查与 CI

提交到 GitHub 的 CI 检查源码编译与消融 YAML 配置一致性，但不能替代真实 GPU 前向、反向传播和解码测试。正式 500 Epoch 训练前应确认：

- 当前分支和 commit SHA 与计划一致。
- 训练数据和开发数据可成功加载，词表大小合理。
- Full 条件实际启用 Thought 和 Routing。
- 两个 No-Thought 条件不实例化 Thought，并关闭 Routing。
- 三个条件的非消融超参数、数据划分、解码参数相同。
- 日志能正常写入，输出目录不覆盖历史运行。
- GPU 显存足以容纳 batch size 32。

可先执行已有静态检查：

```bash
python -m compileall -q main
python -m unittest tests.test_ablation_configs
```

静态检查通过不代表已完成 GPU 训练验收。若短跑检查需要临时修改 Epoch 或 Batch size，请使用独立的临时配置，并且不要把临时检查结果作为正式实验结果。

## 9. 实验记录模板

每次运行至少记录以下字段，建议保存为 CSV 或 JSON：

| 字段 | 记录内容 |
|---|---|
| Variant | full / no_thought / no_thought_dual_evidence |
| Seed | 42 / 123 / 2026 |
| Branch | Git 分支名 |
| Commit | 完整 commit SHA |
| Config | 使用的 YAML 路径 |
| Environment | Python、PyTorch、CUDA、GPU |
| Dataset/features | 数据版本、特征包来源及校验信息 |
| Best step / epoch | 最佳开发集 Checkpoint 对应位置 |
| Dev BLEU-4 | 最佳验证指标 |
| Test BLEU-4 | 最佳 Checkpoint 的测试指标 |
| ROUGE | 若实验评估流程提供 |
| Params / latency / memory | 参数量、推理延迟、峰值显存（如已测量） |
| Notes | W&B、异常、重启或其他环境差异 |

最终报告应分别给出每个随机种子的指标，以及三个种子的均值和标准差。不要只报告最好的一次运行。

## 10. 基准数值的解释边界

所提供的稳定基准包记录：在其对应的上游源码锚点与环境中，三个 Full 基准种子的最佳 Dev BLEU-4 均不低于 25；Test BLEU-4 分别为 27.65、26.81、26.82，均值为 27.09。

这些数字是用户提供的稳定基准参考值，不是当前 `Picodet/SignThought` 消融分支已经取得的实验结果，也不是对不同源码、硬件或环境的性能保证。只有完成实际训练和评估后，才能填写本轮消融结果。

## 11. Main 分支与变更管理

消融开发和训练配置应继续留在 `ablation/*` 分支，不要直接向 `main` 提交实验改动。每次正式运行前记录分支和 commit SHA。合并前应通过 Pull Request 审查源码差异、配置差异和测试结果。

注意：GitHub API 查询时 `main` 的 `protected` 状态为 false。未能启用平台级保护规则前，“不直接修改 main”只是工作流程约定，并不等于 GitHub 已强制阻止直推或强制推送。仓库管理员仍需在 GitHub 设置中启用相应分支保护。
