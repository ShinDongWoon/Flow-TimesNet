# Recursive-TimesNet
**FFT-guided period selection + 2D Inception CNN + embedding-aware adaptive probabilistic head** for multivariate time-series forecasting.  
Keeps the canonical `[B, T, N] → [B, H, N]` interface while adding robust contextualization and calibrated uncertainty.

> **Benchmark highlight** — On the **Walmart (Kaggle)** dataset, this repository achieves **sMAPE ≈ 0.14** (7-day horizon, seeded), demonstrating strong predictive power in a real retail demand setting.

---

## TL;DR
- **Beyond vanilla TimesNet.** We preserve TimesNet’s FFT-to-2D idea but add: channel-robust *shared* period search, **PeriodGrouper** (merge penalized/duplicate periods), **rich embeddings** (IDs + statics + low-rank temporal context), and an **adaptive probabilistic head** (Negative Binomial with dispersion floors) for stability.
- **CNN synergy.** Embeddings lift semantic signal; the 2D Inception CNN captures intra-/inter-period structure; the probabilistic head calibrates rate/dispersion per series and horizon — *three gears meshing* for accuracy and robustness.
- **Direct & recursive.** Train once, forecast direct (`H` at once) or recursive (rolling one-step) with the same checkpoints.
- **Modular data I/O.** Input schemas and test loaders are fully pluggable. Swap CSV layouts, feature sets, and eval folds via config — no code surgery required.

---
# Low-Rank Temporal Context: Turning Static Embeddings into Time-Varying Signals

> **Why this matters**  
> Injecting *static* per-series information (IDs, categories) as a *dynamic* signal that evolves over time is notoriously hard.  
> **`LowRankTemporalContext`** solves this by **low-rank approximation**: a compact, mathematically principled way to compose per-ID temporal “background music” from a tiny set of shared basis waves.

---

## 1) Problem → Why naive approaches break

We want to add to each series \(X_{t,n}\) a learned, time-varying context \(S_{t,n}\), producing
\[
X'_{t,n} \;=\; X_{t,n} \;+\; S_{t,n}\,,
\]
where:
- \(t \in \{0,\dots,L-1\}\): time,
- \(n \in \{1,\dots,N\}\): series ID.

**Naive plan:** learn every \(S_{t,n}\) separately.  
**Cost:** \(L\times N\) parameters (per batch), brittle and overfits.

| Approach | Parameter scale | Generalization | Notes |
|---|---:|---|---|
| Per-timestep per-ID table | \(O(LN)\) | ❌ poor | Memorizes, no sharing |
| One shared temporal vector | \(O(L)\) | ❌ ignores ID | Misses heterogeneity |
| **Low-rank (ours)** | **\(O(LR + NR)\)** | ✅ strong | Shares *basis*, personalizes by *coeffs* |

With rank \(R \ll L\), we reduce cost by orders of magnitude while keeping ID-specific variation.

---

## 2) Key idea → Low-rank temporal factorization

Assume each per-ID context is a linear mix of a **small** set of shared basis signals \(\{b_r(t)\}_{r=1}^R\):
\[
\boxed{ \; S_{t,n} \;\approx\; \sum_{r=1}^{R} w_{n,r}\, b_r(t) \;} \quad\Longleftrightarrow\quad
\underbrace{\mathbf{S}}_{L\times N} \;\approx\; \underbrace{\mathbf{B}}_{L\times R}\;\underbrace{\mathbf{W}}_{R\times N}.
\]

- \(\mathbf{B}\): **shared** bases across IDs (temporal atoms),
- \(\mathbf{W}\): **per-ID** mixture weights,
- \(R\): rank (number of bases), a small hyperparameter.

This is classic **low-rank approximation**: capture the “essence” with a few orthogonal directions.

---

## 3) Our construction → DCT-like cosine bases + zero-mean stabilization

### 3.1 Cosine basis (DCT-II form)
For \(t=0,\dots,L-1\), \(r=1,\dots,R\):
\[
b_r(t) \;=\; \cos\!\Big(\frac{\pi}{L}\,(t+\tfrac{1}{2})\,r\Big).
\]
**Why cosine/DCT?**
- **Energy compaction:** few low-frequency components explain most natural signals.
- **Near-orthogonality:** disentangles components and eases optimization.
- **FFT-friendly:** fast to generate, stable numerics.

### 3.2 Zero-mean constraint (optional, recommended)
We center each basis columnwise:
\[
\tilde{\mathbf{B}} \;=\; \mathbf{B} \;-\; \text{mean}_t(\mathbf{B})\,.
\]
Any linear combo stays near zero-mean:
\[
\text{mean}_t(\tilde{\mathbf{B}}\,\mathbf{W}) \approx \mathbf{0}.
\]
**Effect:** the context **modulates patterns** without drifting the global scale/level of \(X\).

> [!TIP]
> Use zero-mean contexts when you want to **shape temporal texture** (seasonality, pulse, curvature) but **not** bias levels.  
> Turn it off if you explicitly want the context to adjust baselines.

---

## 4) From embeddings to per-ID mixtures

Each series has a static embedding \(\mathbf{e}_n \in \mathbb{R}^{d}\) (category, location, menu, …).  
We predict mixture coefficients via a small head:
\[
\boxed{ \;\mathbf{w}_n \;=\; \text{Linear}(\mathbf{e}_n)\; \in \mathbb{R}^{R}. \;}
\]
- Minimal, fast, and expressive enough to map “who you are” → “how your context sounds.”

---

## 5) Batched synthesis in PyTorch (shape-safe)

Let:
- `basis` \(\in \mathbb{R}^{L\times R}\) (shared),
- `coeff` \(\in \mathbb{R}^{B\times N\times R}\) (from embeddings),
- output `context` \(\in \mathbb{R}^{B\times L\times N}\).

We compute:
```python
# LowRankTemporalContext.forward (conceptual)
context = torch.einsum("lr,bnr->bln", basis, coeff)  # S_{t,n} = sum_r w_{n,r} * b_r(t)
x_out   = x_in + context
```

---

## 6) Complexity & capacity

- **Params:** \(L\times R\) (basis, fixed or learnable) + affine \(d\times R\) (coeff head).  
- **FLOPs:** \(O(B\,L\,N\,R)\) (einsum), linear in rank \(R\).  
- **Memory:** \(\mathbf{B}\) is shared; \(\mathbf{W}\) computed on the fly from embeddings.

| Knob | Role | Rule of thumb |
|---|---|---|
| Rank \(R\) | richness of temporal palette | start \(R\in[4,16]\); increase if underfitting |
| Zero-mean | level-stability | on for pattern-only; off to adjust baselines |
| Basis type | prior over shapes | Cosine/DCT for smoothness; Spline/learned for flexibility |
| Coeff head | ID→mix mapping | Linear is robust; MLP if highly nonlinear IDs |

---

## 7) How it integrates with TimesNet/CNN backbones

`LowRankTemporalContext` **pre-conditions** the input with ID-aware, time-varying structure:

```mermaid
flowchart LR
  E[Static ID Embedding e_n] --> H[Coeff Head (Linear/MLP)]
  H -->|R weights per ID| W((W ∈ ℝ^{R×N}))
  B[(Cosine Basis B ∈ ℝ^{L×R})] --> S{{Einsum: "lr,bnr→bln"}}
  W --> S
  X[Original Series X ∈ ℝ^{B×L×N}] --> A[Add]
  S --> A
  A --> C[TimesNet / CNN Blocks]
  C --> Y[Forecast]
```

---

## 8) Math summary (all in one place)

\[
\begin{aligned}
&\textbf{Goal: } X'_{t,n} = X_{t,n} + S_{t,n}, \quad S_{t,n} \approx \sum_{r=1}^{R} w_{n,r} b_r(t).\\[4pt]
&b_r(t) = \cos\!\Big(\tfrac{\pi}{L}(t+\tfrac{1}{2})r\Big), \qquad
\tilde{\mathbf{B}} = \mathbf{B} - \text{mean}_t(\mathbf{B}).\\[4pt]
&\mathbf{w}_n = \mathbf{W}(\mathbf{e}_n) \in \mathbb{R}^{R} \quad \text{(linear or MLP head).}\\[4pt]
&\Rightarrow\; \mathbf{S} \approx \tilde{\mathbf{B}}\;\mathbf{W} \;\;\Longleftrightarrow\;\;
S_{t,n} \approx \sum_{r=1}^{R} w_{n,r} \tilde{b}_r(t).
\end{aligned}
\]

---

## 9) API sketch (reference)

```python
class LowRankTemporalContext(nn.Module):
    def __init__(self, length: int, rank: int, embed_dim: int,
                 zero_mean: bool = True, learn_basis: bool = False):
        super().__init__()
        self.length, self.rank = length, rank
        # (A) Shared temporal basis B ∈ ℝ^{L×R}
        B = self._cosine_basis(length, rank)  # DCT-II grid
        if zero_mean:
            B = B - B.mean(dim=0, keepdim=True)
        self.basis = nn.Parameter(B, requires_grad=learn_basis)

        # (B) ID-embedding → R weights
        self.context_coeff = nn.Linear(embed_dim, rank)

    @staticmethod
    def _cosine_basis(L, R):
        t = torch.arange(L).float().unsqueeze(1)        # [L,1]
        r = torch.arange(1, R+1).float().unsqueeze(0)   # [1,R]
        return torch.cos(math.pi / float(L) * (t + 0.5) * r)  # [L,R]

    def forward(self, x: torch.Tensor, id_embed: torch.Tensor):
        """
        x:        [B,L,N]  (time-major)
        id_embed: [B,N,D]  (static embeddings per series)
        """
        coeff = self.context_coeff(id_embed)            # [B,N,R]
        context = torch.einsum("lr,bnr->bln", self.basis, coeff)
        return x + context, context
```

---

## 10) Practical guidance

- **Initialization:** keep `learn_basis=False` first; let the linear head learn mixtures on a *fixed* palette.  
- **Regularization:** mild weight decay on the coeff head; optional \(\ell_2\) on coefficients to avoid over-energetic contexts.  
- **Sanity checks:**  
  - Plot a few \(\{b_r(t)\}\) and sampled \(S_{t,n}\).  
  - Verify `mean_t(S_{t,n})≈0` when zero-mean is on.  
  - Ablate \(R\): look for diminishing returns.  
- **When to increase \(R\):** multi-scale seasonality (daily/weekly), heterogeneous venues/menus, long horizons.  
- **When to learn the basis:** if domain rhythms differ from cosines (e.g., holiday pulses, regime switches). Consider adding piecewise or spline bases.

---

## 11) Empirical playbook (repro-friendly)

1. **Baseline:** model w/o context.  
2. **+ Low-rank context:** \(R=8\), zero-mean on, fixed DCT basis.  
3. **Rank sweep:** \(R\in\{4,8,12,16\}\).  
4. **Basis ablation:** fixed DCT vs learnable basis.  
5. **Downstream impact:** check sMAPE/NLL deltas, especially on sparse/volatile IDs.

| Setting | sMAPE ↓ | NLL ↓ | Notes |
|---|---:|---:|---|
| Baseline |  —  |  —  | reference |
| + LowRank (R=8) | **↓** | **↓** | best cost/benefit |
| + Learnable basis (R=8) | ↓ | ↓ | may improve, watch overfit |
| + High rank (R=32) | ~ | ~ | risk: diminishing returns |

---

## 12) Why this is *not* a hack but a principle

1. **Parsimony:** replaces \(O(LN)\) free knobs with \(O(LR+NR)\) structured ones.  
2. **Inductive bias:** smooth, near-orthogonal atoms match the physics of seasonal/slow dynamics.  
3. **Composability:** cleanly adds to any sequence backbone (TimesNet, CNNs, Transformers).  
4. **Controllability:** zero-mean switch separates *pattern* shaping from *level* shifting.

---

## 13) FAQ

- **Q:** Does zero-mean context limit expressivity?  
  **A:** It limits *level* drift, not temporal shape. Turn it off if you want baseline shifts.

- **Q:** How is this different from adding positional encodings?  
  **A:** Positional encodings are shared across IDs; here, **ID-conditioned mixtures** create distinct *per-ID* temporal textures.

- **Q:** Why not just concatenate embeddings to inputs?  
  **A:** Concatenation leaves the network to invent time-varying use of a static vector. Low-rank context **explicitly** converts ID info into a shaped temporal signal, reducing burden on later layers.

---

## 14) One-screen summary

\[
\boxed{
\begin{aligned}
X'_{t,n} &= X_{t,n} + \underbrace{\sum_{r=1}^R w_{n,r}\, \tilde{b}_r(t)}_{\text{ID-conditioned low-rank context}}\\
\tilde{b}_r(t) &= \cos\!\Big(\tfrac{\pi}{L}(t+\tfrac{1}{2})r\Big) - \text{mean}_t(\cdot)\\
\mathbf{w}_n &= \text{Linear}(\mathbf{e}_n)
\end{aligned}}
\quad\Longrightarrow\quad
\text{compact, stable, and ID-aware temporal enrichment.}
\]

---

## What’s New (Core Behavior Updates)
- **Strict validation windowing.** The **validation holdout period must be at least `input_len + pred_len` days**. The trainer enforces this and raises a clear error otherwise.
- **Experiment lineage.** Training emits a `artifacts/metadata.json` (**`meta_version=1`**) capturing window configuration, detected schema, time-feature settings, and static-feature names. The prediction CLI **validates metadata** to surface configuration drift between train and inference.
- **Submission format.** Submissions now write **actual business dates** in the first column (no opaque row keys).
- **Early stopping (optional).** Set `train.early_stopping_patience: <int>` to stop when **validation NB NLL** fails to improve for that many consecutive epochs. Leave unset or `null` to disable.
- **Simple data augmentation.** Enable `data.augment`:
  - `add_noise_std`: std-dev of Gaussian noise added to input windows.
  - `time_shift`: maximum random shift (in steps) applied to window start.
- **Normalization defaults.** `preprocess.normalize: "none"` by default — the model learns on the **original scale**. If disabled, the **saved scaler artifact stores `None`**.
- **Probabilistic forecasts.** The head predicts **Negative Binomial (`rate`, `dispersion`)** and training optimizes **NB-NLL** with `train.min_sigma` (or a learned vector) supplying dispersion floors for stability. **Validation selects by mean NB-NLL** while still reporting **sMAPE** as a secondary diagnostic.
- **Config renames & backward compatibility.**
  - `model.inception_kernel_set` → **`model.kernel_set`** (old name still accepted; a deprecation warning is logged).
  - `model.bottleneck_ratio` controls 1×1 → k×k → 1×1 bottlenecks in each Inception branch.  
    *Ratio > 1* shrinks mid-channels; *Ratio < 1* expands; **1.0** recovers a single k×k convolution (no bottleneck).
- **CUDA graphs & compilation.**
  - Enabling **`train.cuda_graphs: true`** disables **dropout** (model placed in eval mode during capture) trading regularization for throughput.
  - **`train.use_checkpoint`** toggles activation checkpointing (↓ memory, ↑ time) and is **auto-disabled under CUDA graphs**.
  - **Mutual exclusivity:** manual CUDA graphs (`train.cuda_graphs`) and **`train.compile`** (TorchDynamo) are **mutually exclusive** — compiled modules cannot be safely re-captured.
- **Determinism.** **`train.deterministic: true`** seeds all RNGs, disables cuDNN benchmarking, and enables `torch.use_deterministic_algorithms`, yielding reproducible metrics/weights suitable for CI.
- **Window semantics.**
  - Sliding windows are **fixed** to `model.input_len` with **no zero padding**. Both training and inference must provide at least `input_len` steps (temporal covariates may accompany the values tensor).
  - The model **“telescopes”** inputs: `TimesNet.forward` always **crops to the first `input_len` steps** of the periodic canvas, so passing extra history at inference yields the **same `[B, pred_len, N]`** shape as in training.

---

## Why This Differs From “Vanilla” TimesNet

**Vanilla TimesNet (paper-style):** reshape a 1D series into a 2D period-phase grid chosen by **FFT**, then apply **2D CNN** blocks to capture **intra-period** (phase) and **inter-period** (cycle) patterns.

**This repo extends that design** in three principled ways:

1. **Embedding-aware context**
   - **Value + positional + (optional) time features** with configurable normalization (LayerNorm/RMSNorm/decoupled).
   - **Series ID embeddings** and **static covariates** projected/fused into a **context vector**.
   - **Low-Rank Temporal Context (LRTC)** injects a compact learned basis over time so static info can modulate temporal dynamics.

2. **Robust, shared FFT period search**
   - **Channel-median** magnitude with **batch averaging**, **DC removal**, and **log-penalty** for long periods to avoid spurious peaks.
   - **PeriodGrouper** merges near-duplicate periods (log buckets, min-cycle guards), producing stable, *soft-weighted* candidates for CNN processing.

3. **Adaptive probabilistic head**
   - **Negative Binomial (`rate`, `dispersion`)** with `min_sigma`/per-series floors to keep dispersion positive on sparse demand.
   - Heads are AMP-safe; outputs are masked at invalid points.

**Net effect:** Embeddings raise *semantic SNR*, **2D Inception CNN** exploits *phase-by-cycle* structure, and the **probabilistic head** adapts level/dispersion per series — a synergistic trio that outperforms naively stacked modules.

---
## Architecture Overview

- **DataEmbedding**: value + positional + optional time features; integrates **ID & static** embeddings and **LRTC**.  
- **FFTPeriodSelector**: channel-robust spectrum summary → top-k frequencies (DC removed, long-period damped) → period lengths (≥2 cycles).  
- **PeriodGrouper**: merges close periods, yields logits for **softmax weighting**.  
- **TimesBlock (2D Inception CNN)**: reshape `[B,T,N]` to period grids, apply **multi-kernel Inception** with bottlenecks, compute residuals, then **weighted sum across periods**.  
- **Forecast head**: time projection to horizon `H`, plus **Negative Binomial** rate/dispersion heads with stability floors.
- **Training**: NB-NLL (**default**) with AMP-safe masking; supports **direct** and **recursive** decoding; logs **sMAPE/NLL** and coverage.

---

## Installation

```bash
# Python ≥ 3.10; PyTorch ≥ 2.1 recommended
pip install -r requirements.txt
# Optional: CUDA/cuDNN for GPUs; AMP is supported
```
---

## Quickstart (CLI)

```bash
Initiate(Colab)

!git clone https://github.com/ShinDongWoon/Recursive-TimesNet.git
%cd Recursive-TimesNet
!pip install -r requirements.txt
!pip install -e .

# Train
python -m timesnet_forecast.cli train \
  --config configs/default.yaml \
  --override train.lr=1e-3 window.input_len=336 window.pred_len=24

# Predict (direct or recursive; controlled by config)
python -m timesnet_forecast.cli predict --config configs/default.yaml

# Hyperparameter search (Optuna)
python -m timesnet_forecast.cli tune \
  --config configs/default.yaml \
  --space  configs/search_space.yaml
```
---
## Recursive-TimesNet Dataset Usage Guide

### 1. Training Data (`data/train.csv`)
- **Structure**: A single CSV file consisting of three columns: `영업일자` (business date), `영업장명_메뉴명` (store-menu identifier), and `매출수량` (sales quantity).
- Each row represents the sales quantity for a specific (date, store-menu combination), providing the date in `YYYY-MM-DD` format and an integer sales quantity.
- The file is saved in **UTF-8 with BOM**, so BOM handling should be considered when reading it in environments like Python.

### 2. Evaluation Data (`data/test/TEST_*.csv`)
- Consists of **10 files in total**, from `TEST_00.csv` to `TEST_09.csv`, all using the same schema.
- The column structure is identical to the training data: `영업일자`, `영업장명_메뉴명`, and `매출수량`.
- Each file contains **5,404 rows** (193 store-menu combinations × 28 days of records), providing the most recent 4 weeks of sales history for the subsequent 7-day forecast.
- For analysis, you can read a single test file and sort it by the required store-menu combination and date to preprocess it in the same manner as the training data.
 
### 3. Submission Format (`data/sample_submission.csv`)
- Composed of **194 columns in total**. The first column is `영업일자` (e.g., `TEST_00+1day`), and the following 193 columns correspond to each store-menu combination.
- It consists of **70 rows**, where you must fill in 7 days of predictions (`+1day` to `+7day`) for each test set from `TEST_00` to `TEST_09`.
- The sample submission file is also saved in **UTF-8 with BOM**. You should overwrite the `매출수량` prediction values while maintaining the same encoding for submission.
- The model's output must be non-negative sales quantity predictions, and the column order and headers must **exactly match** the sample submission file.

## Data & I/O Modularity

### Pluggable components
- **Schema**: map your `date`, `target`, `series_id` columns via config (auto-infer candidates if unspecified).
- **Loaders**: swap **train CSV**, **test directory** (e.g., `TEST_00.csv … TEST_09.csv`), and **sample submission** without code changes.
- **Features**: enable/disable **calendar**, **holiday**, **Fourier**, **lag/rolling**, and **intermittency** features.
- **Evaluation**: choose **holdout** or **rolling CV**, horizon **H**, and **sMAPE/NLL** aggregation rules.
- **Augmentation**: add **Gaussian noise** and/or **time shifts** to input windows via `data.augment`.


### Validation windowing
- Your **validation holdout** must span **at least `input_len + pred_len` days**.  
  This ensures each eval window has enough history and produces a full horizon.


### Metadata contract
- Training writes `artifacts/metadata.json` (**`meta_version=1`**) with:
  - window sizes (`input_len`, `pred_len`),
  - inferred schema (date/target/id),
  - enabled time-feature set,
  - names of static features aligned to `series_ids`.
- The prediction CLI **compares** the runtime config with this metadata and **fails fast** on drift (mismatched schema, window sizes, or feature toggles).


### Submission output
- The first column is now **the business date column (`submission.date_col`, default `영업일자`)**, not an abstract row key.
  Downstream graders and dashboards can join on calendar directly.


### Normalization
- **Default**: `preprocess.normalize: "none"`.  
- If no scaler is used, the saved scaler artifact is **`None`** and the pipeline stays on **original units**.


### Sliding windows & telescoping
- Windows are exact length **`input_len`**, **no zero-padding**.
- The forward pass **crops** to the first `input_len` steps, so extra history at inference **does not change** output shape.


---

## Programmatic use

~~~python
from timesnet_forecast.config import PipelineConfig
from timesnet_forecast.train import train_once
from timesnet_forecast.predict import predict_once

cfg = PipelineConfig.from_files(
    "configs/default.yaml",
    overrides={"window.input_len": 336, "window.pred_len": 24},
)
val_nll, artifacts = train_once(cfg)  # (best_nll, paths for checkpoints/scalers/schema/etc.)
submission_path = predict_once(cfg)  # CSV written to submission.output_path/out_path
~~~


---

## Configuration Anatomy

~~~yaml
data:
  train_csv: "data/train.csv"
  test_dir: "data/test"
  sample_submission: "data/sample_submission.csv"
  date_col: "영업일자"
  target_col: "매출수량"
  id_col: "영업장명_메뉴명"
  fill_missing_dates: true
  augment:
    add_noise_std: 0.005
    time_shift: 2

preprocess:
  normalize: "none"
  normalize_per_series: true
  clip_negative: true

train:
  device: "cuda"
  epochs: 70
  early_stopping_patience: 5
  batch_size: 128
  lr: 1.0e-4
  amp: true
  cuda_graphs: false
  compile: false
  val:
    strategy: "rolling"
    holdout_days: 35
    rolling_folds: 3
    rolling_step_days: 14

model:
  mode: "direct"
  input_len: 28
  pred_len: 7
  d_model: 128
  d_ff: 256
  n_layers: 2
  k_periods: 2
  kernel_set:
    - [3, 3]
    - [5, 5]
    - [7, 7]
  bottleneck_ratio: 4.0
  id_embed_dim: 32
  static_proj_dim: 32
  static_layernorm: true

window:
  input_len: 28
  pred_len: 7
~~~


### Mutual exclusivity & side-effects
- `train.cuda_graphs: true` → model is captured in **eval mode**; **dropout disabled**.
- `train.use_checkpoint: true` → reduces memory, slower; **auto-disabled** when `cuda_graphs: true`.
- `train.compile: true` **cannot** be combined with `cuda_graphs: true`.


### Determinism
- `train.deterministic: true` seeds RNGs, disables cuDNN benchmarking, and enables deterministic algorithms — ideal for integration tests and CI.


---

## Static Features & Series ID Embeddings

### Automatic statics
- During training, the pipeline computes **simple statics per series** via `compute_series_features` and stores them in the scaler artifact (or alongside it if normalization is disabled).  
- These are **aligned** with `series_ids`, so inference can **reuse** them without recomputation.

### Custom statics
- Supply your own statics as a precomputed tensor **`[num_series, feature_dim]`** and pass them through data-loader hooks in `train_once` (see `series_static` arguments to `_build_dataloader`).

### Interfaces accepted by the model
- `series_ids`: per-series integer identifiers (for **ID embedding**).  
- `series_static`: optional static covariates **`[num_series, feature_dim]`**.

### Hyper-parameters
- `model.id_embed_dim` (default **32**) — width of the learned ID embedding. **Set `0` to disable** when IDs lack signal.  
- `model.static_proj_dim` (default **32**) — projection width applied to static covariates before concatenation; **`null`** keeps raw dimensionality.  
- `model.static_layernorm` — toggles a LayerNorm after the static projection (recommended when mixing disparate scales).

### Memory budgeting
- ID embedding params ≈ `num_series × id_embed_dim`  
  - **Example**: `1,000` series × `64` → **64,000** parameters.
- Static projection params ≈ `static_input_dim × static_proj_dim + static_proj_dim`.  
  Plan GPU memory when increasing these knobs or sweeping them with Optuna.

### CLI overrides & tuning
~~~bash
# Override via CLI
timesnet-forecast train --override model.id_embed_dim=16 model.static_proj_dim=null
~~~

~~~yaml
# Optuna search-space example (search_space.yaml)
model.id_embed_dim:
  type: categorical
  choices: [0, 16, 32]
~~~


---

## Why Embeddings + CNN + Adaptive Head Work (the synergy)

- **Embeddings → raise semantic SNR**
  - ID embeddings disambiguate per-series regimes.
  - Static/meta features tilt the representation toward series-specific baselines.
  - LRTC lets static info modulate temporal filters with a tiny rank budget.

- **2D Inception CNN → structured expressivity**
  - Period-phase grids expose intra-/inter-period patterns.
  - Multi-kernel branches act like “band-pass microscopes” over phase and cycle.

- **Adaptive Probabilistic Head → calibrated outputs**
  - Negative Binomial (`rate`, `dispersion`) forecasts by default for count-style demand.
  - Softplus + dispersion floors (driven by `train.min_sigma` or per-series buffers) stabilize training at low signal levels.

- **Together**
  - Context sets the playing field; CNN plays the structure; the head keeps score honestly.


---

## Losses & Metrics

**Negative-Binomial NLL (default)**
Training and validation call `negative_binomial_nll(rate, dispersion, y, mask)` with masks from `negative_binomial_mask` so only finite targets contribute. Dispersion floors (from `train.min_sigma` or per-series vectors) keep the likelihood well-behaved.

**sMAPE (reported)**
`sMAPE = mean( 2|y - ŷ| / (|y| + |ŷ| + ε) )` over valid targets.


---

## Direct vs. Recursive Forecasting
- **Direct**: single forward pass yields `H`-step forecasts (lower error accumulation).
- **Recursive**: repeated 1-step predictions rolled over horizon (more flexible with covariates).
- Switch via config: set `model.mode: direct` or `model.mode: recursive` (can override via `--override model.mode=recursive`).


---

## Troubleshooting & Tips
- **Periods look unstable?**
  - Increase `k_periods` slightly and enable log-bucket merging.
  - Ensure `min_cycles ≥ 2` and DC removal is on.
- **Over/under-dispersion or poor calibration?**
  - Adjust `train.min_sigma` / `min_sigma_scale`; enrich statics/ID embeddings; review normalization choices.
- **Intermittent zeros dominate?**
  - Stay with the NB head but add calendar covariates or richer statics to stabilize baselines.
- **AMP overflow/NaNs?**
  - Clamp logits/residuals; keep Softplus β modest; verify mixed-precision safe ops.
- **Memory pressure?**
  - Enable `train.use_checkpoint`; reduce `d_model/d_ff`; shrink `kernel_set`; raise `bottleneck_ratio`.
- **Reproducibility?**
  - Set `train.deterministic: true`; fix seed; pin versions; avoid `cuda_graphs` under variant kernels.


---

## Benchmark (Walmart Kaggle)
- **Dataset**: Walmart retail demand (Kaggle)  
- **Split**: rolling CV (7-day horizon), seeded  
- **Metric**: sMAPE (reported), NB NLL (selection)
- **Score**: ≈ **0.14**  
- **Notes**: Negative Binomial head (rate/dispersion) with dispersion floors; embeddings + LRTC on; PeriodGrouper enabled; AMP on GPU.

_Reproduce by fixing `seed`, `window.pred_len=7`, keeping the NB head active with `train.min_sigma`, and using the provided rolling CV profile in `configs/benchmarks/walmart.yaml`._


---

## Internals (Selected)
- **FFTPeriodSelector (shared across channels)**  
  Use `torch.fft.rfft` with channel-median and batch averaging. Remove DC, apply log-penalty on long indices, top-`k_periods`, map to periods, and require `≥ 2` cycles.
- **TimesBlock**  
  Gridify per period, run Inception branches with bottlenecks, compute residuals, softmax-weight across grouped periods, and fuse with skip connections.
- **Performance features**  
  Activation checkpointing (toggle): ↓ memory at ↑ time; auto-off with CUDA graphs.  
  CUDA graphs (toggle): throughput wins; disables dropout; avoid with `train.compile`.  
  `channels-last` and AMP are supported end-to-end.


---

## Roadmap
- Channel-specific period sub-selection (per-series attention over FFT bins)  
- Lightweight exogenous encoder for promotions/events  
- Quantile and CRPS training objectives  
- Dynamic routing across Inception branches


---

## Acknowledgements
This project is built upon the foundational concepts and architecture introduced in the original TimesNet paper. The core implementation of the TimesNet model is inspired by the official source code provided by the authors.

- Original Paper: Haixu Wu, et al. “TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.” ICLR 2023.  
- Official Repository: https://github.com/thuml/TimesNet  
- The original TimesNet source code is licensed under the **Apache License 2.0**. A copy of the license can be found in the **NOTICE** file within this repository.
