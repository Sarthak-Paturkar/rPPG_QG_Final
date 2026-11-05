import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import (
    PREPROCESSING_BAND_HZ,
    PREPROCESSING_DETREND_TYPE,
    PREPROCESSING_FILTER_ORDER,
    PREPROCESSING_MAX_INTERPOLATION_GAP_SECONDS,
    PREPROCESSING_MIN_SKIN_RATIO,
    PREPROCESSING_NORMALIZATION,
    PREPROCESSING_TIMESTAMP_RELATIVE_TOLERANCE,
)
from scripts.run_phase_a_b_batch import (
    ROI_NAMES,
    _output_path_map,
    _persist_summary,
    _subject_paths,
    validate_phase_a_outputs,
    validate_phase_b_outputs,
    validate_rgb_output,
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test")


def _source_dataframe() -> pd.DataFrame:
    rows = 60
    data: dict[str, np.ndarray] = {
        "frame": np.arange(rows),
        "video_time_seconds": np.arange(rows) / 30.0,
        "face_detected": np.ones(rows, dtype=int),
        "face_confidence": np.full(rows, 0.95),
    }
    for roi_index, roi in enumerate(ROI_NAMES):
        for channel_index, channel in enumerate(("r", "g", "b")):
            data[f"{roi}_{channel}"] = (
                100 + roi_index * 10 + channel_index + np.sin(np.arange(rows) / 4)
            )
        data[f"{roi}_skin_ratio"] = np.full(rows, 0.9)
    return pd.DataFrame(data)


class BatchOutputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.subject = "subject3"
        self.paths = _subject_paths(self.subject)
        self.source_df = _source_dataframe()
        self.rgb_path = self.root / "rgb_signals.csv"
        self.source_df.to_csv(self.rgb_path, index=False)
        self.paths["rgb_csv"] = self.rgb_path
        self.paths["phase_a_summary"] = self.root / "reports" / "summary.json"
        self.paths["phase_a_report_dir"] = self.root / "reports"
        self.paths["phase_a_plot_dir"] = self.root / "plots" / "quality"
        self.paths["phase_b_csv"] = self.root / "cleaned.csv"
        self.paths["phase_b_report"] = self.root / "reports" / "preprocessing.json"
        self.paths["phase_b_plot_dir"] = self.root / "plots" / "preprocessing"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_rgb_csv_validation_accepts_expected_schema_and_monotonic_time(self) -> None:
        valid, message, frame = validate_rgb_output(self.rgb_path)
        self.assertTrue(valid, message)
        self.assertEqual(len(frame), len(self.source_df))

        self.source_df.assign(video_time_seconds=0.0).to_csv(
            self.rgb_path, index=False
        )
        valid, message, _ = validate_rgb_output(self.rgb_path)
        self.assertFalse(valid)
        self.assertIn("strictly increasing", message)

    def test_phase_a_requires_reports_to_match_current_rgb_data(self) -> None:
        self.paths["phase_a_report_dir"].mkdir(parents=True)
        self.paths["phase_a_plot_dir"].mkdir(parents=True)
        self.paths["phase_a_summary"].write_text(
            json.dumps(
                {
                    "subject": self.subject,
                    "sample_count": len(self.source_df),
                    "source_csv": str(self.rgb_path),
                    "roi_summary": {
                        roi: {"valid_rgb_sample_count": len(self.source_df)}
                        for roi in ROI_NAMES
                    },
                }
            ),
            encoding="utf-8",
        )
        for roi in ROI_NAMES:
            channels = {}
            for channel in ("r", "g", "b"):
                values = self.source_df[f"{roi}_{channel}"].to_numpy()
                channels[channel] = {
                    "statistics": {
                        "finite_count": int(len(values)),
                        "missing_count": 0,
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "standard_deviation": float(np.std(values, ddof=1)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                }
            (self.paths["phase_a_report_dir"] / f"{roi}_quality.json").write_text(
                json.dumps(
                    {
                        "subject": self.subject,
                        "roi": roi,
                        "sample_count": len(self.source_df),
                        "roi_valid_sample_count": len(self.source_df),
                        "channels": channels,
                    }
                ),
                encoding="utf-8",
            )
            for plot in ("raw_rgb", "derivatives", "outliers"):
                _write_png(self.paths["phase_a_plot_dir"] / f"{roi}_{plot}.png")
        _write_png(self.paths["phase_a_plot_dir"] / "roi_quality_comparison.png")

        valid, message = validate_phase_a_outputs(
            self.subject, self.paths, self.source_df
        )
        self.assertTrue(valid, message)

        self.source_df.loc[0, "forehead_r"] += 1
        valid, message = validate_phase_a_outputs(
            self.subject, self.paths, self.source_df
        )
        self.assertFalse(valid)
        self.assertIn("does not match", message)

    def test_phase_a_accepts_reported_all_missing_channel_data(self) -> None:
        self.source_df["forehead_r"] = np.nan
        self.source_df.to_csv(self.rgb_path, index=False)
        self.paths["phase_a_report_dir"].mkdir(parents=True)
        self.paths["phase_a_plot_dir"].mkdir(parents=True)
        self.paths["phase_a_summary"].write_text(
            json.dumps(
                {
                    "subject": self.subject,
                    "sample_count": len(self.source_df),
                    "source_csv": str(self.rgb_path),
                    "roi_summary": {
                        roi: {
                            "valid_rgb_sample_count": (
                                0 if roi == "forehead" else len(self.source_df)
                            )
                        }
                        for roi in ROI_NAMES
                    },
                }
            ),
            encoding="utf-8",
        )
        for roi in ROI_NAMES:
            channels = {}
            for channel in ("r", "g", "b"):
                values = pd.to_numeric(
                    self.source_df[f"{roi}_{channel}"], errors="coerce"
                ).to_numpy()
                finite = values[np.isfinite(values)]
                channels[channel] = {
                    "statistics": {
                        "finite_count": int(finite.size),
                        "missing_count": len(values) - int(finite.size),
                        "mean": float(np.mean(finite)) if finite.size else None,
                        "median": float(np.median(finite)) if finite.size else None,
                        "standard_deviation": (
                            float(np.std(finite, ddof=1))
                            if finite.size > 1
                            else (0.0 if finite.size == 1 else None)
                        ),
                        "minimum": float(np.min(finite)) if finite.size else None,
                        "maximum": float(np.max(finite)) if finite.size else None,
                    }
                }
            valid_count = int(
                np.logical_and.reduce(
                    [
                        np.isfinite(self.source_df[f"{roi}_{channel}"])
                        for channel in ("r", "g", "b")
                    ]
                ).sum()
            )
            (self.paths["phase_a_report_dir"] / f"{roi}_quality.json").write_text(
                json.dumps(
                    {
                        "subject": self.subject,
                        "roi": roi,
                        "sample_count": len(self.source_df),
                        "roi_valid_sample_count": valid_count,
                        "channels": channels,
                    }
                ),
                encoding="utf-8",
            )
            for plot in ("raw_rgb", "derivatives", "outliers"):
                _write_png(self.paths["phase_a_plot_dir"] / f"{roi}_{plot}.png")
        _write_png(self.paths["phase_a_plot_dir"] / "roi_quality_comparison.png")

        valid, message = validate_phase_a_outputs(
            self.subject, self.paths, self.source_df
        )
        self.assertTrue(valid, message)

    def test_phase_b_requires_preserved_source_and_expected_artifacts(self) -> None:
        self.paths["phase_b_csv"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["phase_b_report"].parent.mkdir(parents=True, exist_ok=True)
        processed = self.source_df.copy()
        for roi in ROI_NAMES:
            for channel in ("r", "g", "b"):
                for stage in ("filled", "detrended", "filtered", "normalized"):
                    processed[f"{roi}_{channel}_{stage}"] = np.zeros(len(processed))
            processed[f"{roi}_valid_sample"] = True
            processed[f"{roi}_interpolated_sample"] = False
            processed[f"{roi}_unfilled_invalid_sample"] = False
        processed.to_csv(self.paths["phase_b_csv"], index=False)
        self.paths["phase_b_report"].write_text(
            json.dumps(
                {
                    "source_row_count": len(self.source_df),
                    "sampling_rate_hz_from_timestamps": 30.0,
                    "timestamp_relative_uniformity_tolerance": (
                        PREPROCESSING_TIMESTAMP_RELATIVE_TOLERANCE
                    ),
                    "settings": {
                        "bandpass_hz": list(PREPROCESSING_BAND_HZ),
                        "butterworth_order": PREPROCESSING_FILTER_ORDER,
                        "minimum_skin_ratio": PREPROCESSING_MIN_SKIN_RATIO,
                        "maximum_interpolated_internal_gap_seconds": (
                            PREPROCESSING_MAX_INTERPOLATION_GAP_SECONDS
                        ),
                        "detrend_type": PREPROCESSING_DETREND_TYPE,
                        "normalization": PREPROCESSING_NORMALIZATION,
                        "source_csv_modified": False,
                    },
                    "rois": {roi: {} for roi in ROI_NAMES},
                }
            ),
            encoding="utf-8",
        )
        for roi in ROI_NAMES:
            _write_png(
                self.paths["phase_b_plot_dir"] / f"{roi}_preprocessing.png"
            )

        valid, message = validate_phase_b_outputs(self.paths, self.source_df)
        self.assertTrue(valid, message)

        processed.loc[0, "forehead_r"] += 1
        processed.to_csv(self.paths["phase_b_csv"], index=False)
        valid, message = validate_phase_b_outputs(self.paths, self.source_df)
        self.assertFalse(valid)
        self.assertIn("altered source column", message)

    def test_expected_artifact_map_covers_all_three_stages(self) -> None:
        paths = _output_path_map(self.paths)
        self.assertEqual(set(paths), {"rgb_extraction", "phase_a", "phase_b"})
        self.assertEqual(len(paths["phase_a"]), 14)
        self.assertEqual(len(paths["phase_b"]), 5)

    def test_summary_persistence_retries_transient_replace_permission_error(self) -> None:
        summary_path = self.root / "batch_summary.json"
        original_replace = os.replace
        attempts = 0

        def replace_with_transient_error(source: Path, destination: Path) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise PermissionError("temporarily locked")
            original_replace(source, destination)

        record = {
            "subject": self.subject,
            "status": "completed",
            "stages": {
                stage: {"status": "pending"}
                for stage in ("rgb_extraction", "phase_a", "phase_b")
            },
        }
        with patch(
            "scripts.run_phase_a_b_batch.os.replace",
            side_effect=replace_with_transient_error,
        ):
            _persist_summary(summary_path, [self.subject], {self.subject: record})

        self.assertEqual(attempts, 2)
        self.assertEqual(
            json.loads(summary_path.read_text(encoding="utf-8"))["status"],
            "completed",
        )
        self.assertEqual(list(self.root.glob(".batch_summary.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
