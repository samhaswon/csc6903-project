"""Tests for the 2020 consumer-grid simulation export helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from run_consumer_grid_2020_simulation import (
    CSV_EXPORT_COLUMNS_TO_DROP,
    build_export_frame,
    build_first_week_export_frame,
)


class BuildExportFrameTest(unittest.TestCase):
    """Verify the final CSV export excludes internal source columns."""

    def test_drops_requested_columns(self) -> None:
        """The export frame should omit every requested column."""
        data = {
            "timestamp_utc": ["2020-01-01T00:00:00Z"],
            "keep_column": [1.0],
        }
        for index, column in enumerate(CSV_EXPORT_COLUMNS_TO_DROP, start=1):
            data[column] = [index]

        frame = pd.DataFrame(data)
        export_frame = build_export_frame(frame)

        self.assertEqual(list(export_frame.columns), ["timestamp_utc", "keep_column"])
        for column in CSV_EXPORT_COLUMNS_TO_DROP:
            self.assertNotIn(column, export_frame.columns)


class BuildFirstWeekExportFrameTest(unittest.TestCase):
    """Verify the first-week export is timestamp filtered."""

    def test_keeps_only_first_seven_days(self) -> None:
        """The first-week export should stop before day eight."""
        frame = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(
                    [
                        "2020-01-01T00:00:00Z",
                        "2020-01-07T23:59:00Z",
                        "2020-01-08T00:00:00Z",
                    ],
                    utc=True,
                ),
                "keep_column": [1.0, 2.0, 3.0],
            }
        )

        export_frame = build_first_week_export_frame(frame)

        self.assertEqual(len(export_frame), 2)
        self.assertEqual(
            list(export_frame["timestamp_utc"]),
            list(pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-07T23:59:00Z"], utc=True)),
        )
        self.assertNotIn(pd.Timestamp("2020-01-08T00:00:00Z", tz="UTC"), export_frame.values)


if __name__ == "__main__":
    unittest.main()
