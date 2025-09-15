# Recursive-TimesNet

- Validation holdout period must be at least `input_len + pred_len` days.
- Submission files now output actual business dates in the first column instead of row keys.
- Optional early stopping: set `train.early_stopping_patience` to an integer to stop training
  when validation WSMAPE does not improve for that many consecutive epochs. Leave it unset or
  `null` to disable early stopping.
- Simple data augmentation can be enabled via the `data.augment` section of the config.
  - `add_noise_std`: standard deviation of Gaussian noise added to input windows.
  - `time_shift`: maximum number of time steps to randomly shift each window's start index.
- Configuration option `model.inception_kernel_set` has been renamed to `model.kernel_set`. The previous name is still accepted for backward compatibility.
- Using CUDA Graphs (`train.cuda_graphs: true`) disables dropout because the model is placed in evaluation mode during graph capture. This trades regularization for faster execution.
- `model.pmax_cap` limits the automatically inferred maximum period length. During training the dominant period across all series is detected and then clipped to this cap to avoid extremely long seasonal windows.
- The model "telescopes" input sequences: `TimesNet.forward` always crops to the last `input_len` steps, so passing extra history at inference produces the same `[B, pred_len, N]` shaped output as training.
