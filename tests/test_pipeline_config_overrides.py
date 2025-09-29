from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.config import PipelineConfig  # noqa: E402


def test_window_override_updates_model_and_window_sections() -> None:
    overrides = [
        "window.input_len=14",
        "window.pred_len=5",
        "window.stride=3",
        "window.short_series_strategy=pad",
        "window.pad_value=2.5",
    ]
    cfg = PipelineConfig.from_files("configs/default.yaml", overrides=overrides)

    assert cfg.window.input_len == 14
    assert cfg.window.pred_len == 5
    assert cfg.window.stride == 3
    assert cfg.window.short_series_strategy == "pad"
    assert cfg.window.pad_value == pytest.approx(2.5)

    cfg_dict = cfg.to_dict()
    assert cfg_dict["model"]["input_len"] == 14
    assert cfg_dict["model"]["pred_len"] == 5
    assert cfg_dict["window"]["stride"] == 3
    assert cfg_dict["window"]["short_series_strategy"] == "pad"
    assert cfg_dict["window"]["pad_value"] == pytest.approx(2.5)
