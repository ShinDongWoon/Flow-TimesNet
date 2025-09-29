# Recursive-TimesNet

- Validation holdout period must be at least `input_len + pred_len` days.
- Training now emits a `metadata.json` artifact (starting at `meta_version=1`) capturing the
  window configuration, detected schema, time feature settings, and static feature names. The
  prediction CLI validates this metadata to surface clear errors when configurations drift from
  the training setup.
- Submission files now output actual business dates in the first column instead of row keys.
- Optional early stopping: set `train.early_stopping_patience` to an integer to stop training
  when the validation Gaussian NLL does not improve for that many consecutive epochs. Leave it unset or
  `null` to disable early stopping.
- Simple data augmentation can be enabled via the `data.augment` section of the config.
  - `add_noise_std`: standard deviation of Gaussian noise added to input windows.
  - `time_shift`: maximum number of time steps to randomly shift each window's start index.
- Normalisation is disabled by default (`preprocess.normalize: "none"`) so the model trains on
  the original data scale. When normalisation is disabled the saved scaler artifact stores `None`.
- TimesNet now emits probabilistic forecasts `(mu, sigma)` and training optimises a Gaussian
  negative log-likelihood with a configurable `train.min_sigma` lower bound for numerical
  stability. Validation uses the mean Gaussian NLL for model selection while still reporting
  SMAPE as a secondary diagnostic metric.
- Configuration option `model.inception_kernel_set` has been renamed to `model.kernel_set`. The previous name is still accepted for backward compatibility.
- `model.bottleneck_ratio` toggles ``1×1 → k×k → 1×1`` bottlenecks inside every inception branch.
  Ratios greater than ``1`` shrink the hidden width relative to ``min(in_ch, out_ch)`` while ratios
  below ``1`` expand it; set the value to ``1.0`` to recover the previous single ``k×k`` convolution
  without bottlenecks.
- Using CUDA Graphs (`train.cuda_graphs: true`) disables dropout because the model is placed in evaluation mode during graph capture. This trades regularization for faster execution.
- Activation checkpointing can be toggled via `train.use_checkpoint`. Enabling it reduces memory usage at the cost of slower training throughput and is automatically turned off when CUDA graphs are active.
- Manual CUDA graph capture (`train.cuda_graphs`) and `torch.compile` (`train.compile`) are mutually exclusive. TorchDynamo already performs graph capture and its compiled modules cannot be re-captured safely.
- Enable fully deterministic execution by setting `train.deterministic: true`. This seeds every RNG, disables cuDNN benchmarking, and enforces deterministic algorithms via `torch.use_deterministic_algorithms`, yielding reproducible metrics and weights for integration tests.
- Sliding windows are fixed to `model.input_len` without zero padding. Ensure both training and inference provide at least that much history; optional temporal covariates can be passed alongside the values tensor when using the built-in embedding.
- The model "telescopes" input sequences: `TimesNet.forward` always crops to the first `input_len` steps of the periodic canvas, so passing extra history at inference produces the same `[B, pred_len, N]` shaped output as training.

## Static features and series ID embeddings

- During training the pipeline computes simple static covariates for every series via `compute_series_features` and stores them in the scaler artifact. These features are aligned with the `ids` array so inference can re-use them without recomputation. Custom static features can be supplied by precomputing a `[num_series, feature_dim]` tensor and passing it through the dataloader hooks in `train_once` (see the `series_static` arguments to `_build_dataloader`).
- `TimesNet` now accepts per-series identifiers via `series_ids` and optional static covariates via `series_static`. The configuration exposes the relevant hyper-parameters:
  - `model.id_embed_dim` (default `32`) controls the width of the learned ID embedding. Set it to `0` to disable the embedding entirely when identifiers are not informative.
  - `model.static_proj_dim` (default `32`) specifies the projection width applied to static covariates before they are concatenated with temporal features. Use `null` to keep the raw feature dimensionality or reduce the value to shrink the projection.
  - `model.static_layernorm` toggles a `LayerNorm` after the static projection. Leaving it enabled stabilises training when mixing features with very different scales.
- Larger embedding dimensions increase both parameter count and activation memory. The ID embedding contributes approximately `num_series × id_embed_dim` parameters (for example, 1 000 series with a width of 64 adds 64 000 weights), while the static projection introduces roughly `static_input_dim × static_proj_dim + static_proj_dim` parameters. Plan GPU memory accordingly when increasing these knobs or when enabling Optuna sweeps over them.
- Override the new hyper-parameters via the CLI, e.g. `timesnet-forecast train --override model.id_embed_dim=16 model.static_proj_dim=null`, or include them in Optuna search spaces (`model.id_embed_dim: {choices: [0, 16, 32], type: "categorical"}`) to tune their impact automatically.

## Acknowledgements

This project is built upon the foundational concepts and architecture introduced in the original TimesNet paper. The core implementation of the TimesNet model is inspired by the official source code provided by the authors.

- **Original Paper**: Haixu Wu, et al. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." *ICLR 2023*.
- **Official Repository**: [https://github.com/thuml/TimesNet](https://github.com/thuml/TimesNet)

The original TimesNet source code is licensed under the Apache License 2.0. A copy of the license can be found in the `NOTICE` file within this repository.
