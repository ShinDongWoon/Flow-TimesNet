from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type

import pandas as pd

from .io import normalize_series_name, parse_row_key


logger = logging.getLogger(__name__)


@dataclass
class SubmissionRowMeta:
    test_part: str
    step: int
    date: pd.Timestamp | None = None
    source: str | None = None


@dataclass
class SubmissionContext:
    predictions_columns: List[str]
    row_meta: Mapping[str, SubmissionRowMeta]
    row_order: List[str]
    test_parts: Mapping[str, Sequence[str]]
    ids: Sequence[str]
    output_order: List[str]
    normalized_to_output: Mapping[str, str]
    sample_df: Optional[pd.DataFrame]
    row_key_column: str
    date_column: str
    default_fill_value: float
    new_ids: Sequence[str]
    missing_ids: Sequence[str]
    missing_by_part: Mapping[str, Sequence[str]]

    @property
    def output_columns(self) -> List[str]:
        return [self.normalized_to_output.get(col, col) for col in self.output_order]


class SubmissionWriter(ABC):
    """Strategy interface for writing submission files."""

    missing_policy: str = "warn_fill"

    def __init__(
        self,
        *,
        default_fill_value: float = 0.0,
        missing_policy: Optional[str] = None,
    ) -> None:
        self.default_fill_value = default_fill_value
        if missing_policy:
            self.missing_policy = str(missing_policy)

    def render(self, predictions: pd.DataFrame, context: SubmissionContext) -> pd.DataFrame:
        self._validate_predictions(predictions, context)
        template = self._prepare_template(context)
        filled = self._fill_template(template, predictions, context)
        self._validate_output(filled, context)
        return filled

    def _validate_predictions(self, predictions: pd.DataFrame, context: SubmissionContext) -> None:
        required = [col for col in context.output_order if col not in context.new_ids]
        missing = [col for col in required if col not in predictions.columns]
        if missing:
            raise ValueError(
                "Predictions missing required columns: " + ", ".join(missing)
            )
        extra = [col for col in predictions.columns if col not in context.output_order]
        if extra:
            logger.debug("Predictions contain extra columns not used in output: %s", extra)

    def _validate_output(self, df: pd.DataFrame, context: SubmissionContext) -> None:
        expected_columns = self._expected_columns(context)
        if list(df.columns) != expected_columns:
            raise ValueError(
                "Submission output columns mismatch; expected "
                f"{expected_columns} but received {list(df.columns)}"
            )
        if len(df) != len(context.row_order):
            raise ValueError(
                f"Submission row count mismatch; expected {len(context.row_order)} rows but received {len(df)}"
            )

    def _expected_columns(self, context: SubmissionContext) -> List[str]:
        return context.output_columns

    def _prepare_template(self, context: SubmissionContext) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _fill_template(
        self,
        template: pd.DataFrame,
        predictions: pd.DataFrame,
        context: SubmissionContext,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def _default_row_values(self, context: SubmissionContext) -> List[float]:
        return [self.default_fill_value for _ in context.output_order]

    def _handle_missing_row(
        self, row_key: str, context: SubmissionContext, reason: str
    ) -> List[float]:
        if self.missing_policy == "error":
            raise KeyError(f"Missing prediction for {row_key} ({reason})")
        logger.warning("Missing prediction for %s (%s); filling defaults", row_key, reason)
        return self._default_row_values(context)


class RowKeyLongWriter(SubmissionWriter):
    """Legacy row-key wide submission writer."""

    def _expected_columns(self, context: SubmissionContext) -> List[str]:
        return [context.row_key_column, *context.output_columns]

    def _prepare_template(self, context: SubmissionContext) -> pd.DataFrame:
        if context.sample_df is not None:
            template = context.sample_df.copy()
        else:
            template = pd.DataFrame({context.row_key_column: context.row_order})
            for column in context.output_columns:
                template[column] = context.default_fill_value
        return template

    def _fill_template(
        self,
        template: pd.DataFrame,
        predictions: pd.DataFrame,
        context: SubmissionContext,
    ) -> pd.DataFrame:
        df = template.copy()
        for idx, raw_value in enumerate(df[context.row_key_column]):
            try:
                part, step = parse_row_key(str(raw_value))
                row_key = f"{part}+D{int(step)}"
            except Exception:  # noqa: BLE001
                values = self._handle_missing_row(str(raw_value), context, "invalid_row_key")
                df.loc[idx, context.output_columns] = values
                continue

            if row_key not in context.row_meta:
                values = self._handle_missing_row(row_key, context, "unknown_row")
                df.loc[idx, context.output_columns] = values
                continue

            if row_key not in predictions.index:
                values = self._handle_missing_row(row_key, context, "missing_prediction")
                df.loc[idx, context.output_columns] = values
                continue

            pred_series = predictions.loc[row_key]
            values = [
                float(pred_series.get(col, self.default_fill_value))
                for col in context.output_order
            ]
            df.loc[idx, context.output_columns] = values

        return df


class DateMenuWriter(SubmissionWriter):
    """Submission writer that uses actual forecast dates in the first column."""

    def _expected_columns(self, context: SubmissionContext) -> List[str]:
        return [context.date_column, *context.output_columns]

    def _prepare_template(self, context: SubmissionContext) -> pd.DataFrame:
        rows: List[Tuple[pd.Timestamp | str, List[float]]] = []
        for row_key in context.row_order:
            meta = context.row_meta.get(row_key)
            date_val = meta.date if meta else row_key
            rows.append((date_val, self._default_row_values(context)))

        data = {
            context.date_column: [row[0] for row in rows],
        }
        for col_idx, column in enumerate(context.output_columns):
            data[column] = [row[1][col_idx] for row in rows]
        return pd.DataFrame(data)

    def _fill_template(
        self,
        template: pd.DataFrame,
        predictions: pd.DataFrame,
        context: SubmissionContext,
    ) -> pd.DataFrame:
        df = template.copy()
        for idx, row_key in enumerate(context.row_order):
            if row_key in predictions.index:
                pred_series = predictions.loc[row_key]
                values = [
                    float(pred_series.get(col, self.default_fill_value))
                    for col in context.output_order
                ]
            else:
                values = self._handle_missing_row(row_key, context, "missing_prediction")

            df.loc[idx, context.output_columns] = values
            meta = context.row_meta.get(row_key)
            if meta and meta.date is not None:
                df.at[idx, context.date_column] = meta.date
            else:
                df.at[idx, context.date_column] = row_key

        return df


WRITER_REGISTRY: Dict[str, Type[SubmissionWriter]] = {
    "date_menu": DateMenuWriter,
    "row_key": RowKeyLongWriter,
    "row_key_long": RowKeyLongWriter,
}


def get_submission_writer(name: str) -> Type[SubmissionWriter]:
    key = (name or "date_menu").lower()
    if key not in WRITER_REGISTRY:
        raise KeyError(f"Unknown submission writer format '{name}'")
    return WRITER_REGISTRY[key]


def build_submission_context(
    *,
    predictions: pd.DataFrame,
    sample_df: Optional[pd.DataFrame],
    row_meta: Mapping[str, SubmissionRowMeta],
    row_order: Sequence[str],
    test_parts: Mapping[str, Sequence[str]],
    ids: Sequence[str],
    new_ids: Sequence[str],
    missing_ids: Sequence[str],
    missing_by_part: Mapping[str, Sequence[str]],
    submission_cfg: Mapping[str, object],
) -> SubmissionContext:
    default_fill_value = float(submission_cfg.get("default_fill_value", 0.0) or 0.0)
    date_column = str(submission_cfg.get("date_col", "date"))
    row_key_column = str(submission_cfg.get("row_key_col", "row_key"))

    if sample_df is not None and not sample_df.empty:
        row_key_column = str(sample_df.columns[0])
        menu_columns = list(sample_df.columns[1:])
        normalized = [normalize_series_name(col) for col in menu_columns]
    else:
        menu_columns = list(ids)
        normalized = [normalize_series_name(col) for col in menu_columns]

    normalized_to_output = dict(zip(normalized, menu_columns))
    output_order = normalized

    context = SubmissionContext(
        predictions_columns=list(predictions.columns),
        row_meta=row_meta,
        row_order=list(row_order),
        test_parts=test_parts,
        ids=list(ids),
        output_order=output_order,
        normalized_to_output=normalized_to_output,
        sample_df=sample_df,
        row_key_column=row_key_column,
        date_column=date_column,
        default_fill_value=default_fill_value,
        new_ids=list(new_ids),
        missing_ids=list(missing_ids),
        missing_by_part=missing_by_part,
    )
    return context
