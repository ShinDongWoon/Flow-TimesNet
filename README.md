# Recursive-TimesNet

- Validation holdout period must be at least `input_len + pred_len` days.
- Submission files now output actual business dates in the first column instead of row keys.
- Optional early stopping: set `train.early_stopping_patience` to an integer to stop training
  when validation WSMAPE does not improve for that many consecutive epochs. Leave it unset or
  `null` to disable early stopping.
