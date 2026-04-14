"""Tests for consumer_grid_web_ui helper behavior."""

from __future__ import annotations

import unittest

import pandas as pd

from consumer_grid_web_ui import _build_week_ranges, _parse_numeric_options


class BuildWeekRangesTest(unittest.TestCase):
    """Verify week range generation from timestamps."""

    def test_builds_only_complete_weeks(self) -> None:
        """Only full 7-day windows should be emitted."""
        timestamps = pd.date_range(
            start="2020-01-01T00:00:00Z",
            end="2020-01-15T00:00:00Z",
            freq="min",
            tz="UTC",
        )
        ranges = _build_week_ranges(pd.Series(timestamps))

        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0][0], pd.Timestamp("2020-01-01T00:00:00Z"))
        self.assertEqual(ranges[0][1], pd.Timestamp("2020-01-08T00:00:00Z"))
        self.assertEqual(ranges[1][0], pd.Timestamp("2020-01-08T00:00:00Z"))
        self.assertEqual(ranges[1][1], pd.Timestamp("2020-01-15T00:00:00Z"))


class ParseNumericOptionsTest(unittest.TestCase):
    """Verify API option parsing and validation."""

    def test_accepts_defaults_when_payload_empty(self) -> None:
        """Missing fields should use configured defaults."""
        options = _parse_numeric_options({})
        self.assertIn("consumer_group_multiplier", options)
        self.assertIn("cheap_price_quantile", options)
        self.assertIn("expensive_price_quantile", options)

    def test_rejects_invalid_quantile_order(self) -> None:
        """Cheap quantile must remain lower than expensive quantile."""
        with self.assertRaisesRegex(ValueError, "cheap_price_quantile"):
            _parse_numeric_options(
                {
                    "cheap_price_quantile": 0.9,
                    "expensive_price_quantile": 0.2,
                }
            )


if __name__ == "__main__":
    unittest.main()
