from pathlib import Path
import sys

# Ensure src path is available for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.utils.io import parse_row_key


def test_parse_row_key_day():
    assert parse_row_key("TEST_00+Day 1") == ("TEST_00", 1)


def test_parse_row_key_korean():
    assert parse_row_key("TEST_00+1일") == ("TEST_00", 1)


def test_parse_row_key_d_prefix():
    assert parse_row_key("TEST_00+D1") == ("TEST_00", 1)
