# Recursive-TimesNet

- Validation holdout period must be at least `input_len + pred_len` days.
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
- Using CUDA Graphs (`train.cuda_graphs: true`) disables dropout because the model is placed in evaluation mode during graph capture. This trades regularization for faster execution.
- Activation checkpointing can be toggled via `train.use_checkpoint`. Enabling it reduces memory usage at the cost of slower training throughput and is automatically turned off when CUDA graphs are active.
- Manual CUDA graph capture (`train.cuda_graphs`) and `torch.compile` (`train.compile`) are mutually exclusive. TorchDynamo already performs graph capture and its compiled modules cannot be re-captured safely.
- `model.pmax_cap` limits the automatically inferred maximum period length. During training the dominant period across all series is detected and then clipped to this cap to avoid extremely long seasonal windows.
- The model "telescopes" input sequences: `TimesNet.forward` always crops to the first `input_len` steps of the periodic canvas, so passing extra history at inference produces the same `[B, pred_len, N]` shaped output as training.
- Training progress is now tracked with Rich-based task groups covering data preparation, model setup, epoch/batch loops, and artifact export. The tracker automatically falls back to plain log messages when stdout is not a TTY so it remains safe for CI logs. Set `train.logging.progress: false` to disable interactive progress reporting when integrating with custom runners.
