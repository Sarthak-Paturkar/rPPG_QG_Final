import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PREPROCESSING_BAND_HZ,
    PREPROCESSING_DETREND_TYPE,
    PREPROCESSING_FILTER_ORDER,
    PREPROCESSING_MAX_INTERPOLATION_GAP_SECONDS,
    PREPROCESSING_MIN_SKIN_RATIO,
    PREPROCESSING_NORMALIZATION,
    PREPROCESSING_TIMESTAMP_RELATIVE_TOLERANCE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROI_NAMES = ("forehead", "right_cheek", "left_cheek")
CHANNELS = ("r", "g", "b")
LOGGER = logging.getLogger("signal_preprocessing")


@dataclass(frozen=True)
class PreprocessingSettings:
    low_cut_hz: float = PREPROCESSING_BAND_HZ[0]
    high_cut_hz: float = PREPROCESSING_BAND_HZ[1]
    filter_order: int = PREPROCESSING_FILTER_ORDER
    min_skin_ratio: float = PREPROCESSING_MIN_SKIN_RATIO
    max_interpolation_gap_seconds: float = (
        PREPROCESSING_MAX_INTERPOLATION_GAP_SECONDS
    )
    timestamp_relative_tolerance: float = (
        PREPROCESSING_TIMESTAMP_RELATIVE_TOLERANCE
    )
    detrend_type: str = PREPROCESSING_DETREND_TYPE
    normalization: str = PREPROCESSING_NORMALIZATION


DEFAULT_SETTINGS = PreprocessingSettings()


@dataclass
class ChannelStages:
    filled: np.ndarray
    detrended: np.ndarray
    filtered: np.ndarray
    normalized: np.ndarray
    interpolated_mask: np.ndarray
    filtered_segment_count: int
    short_segment_count: int
    zero_variance_segment_count: int


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def infer_sampling_rate(
    timestamps: np.ndarray, relative_tolerance: float
) -> tuple[float, float]:
    """Return sampling rate and median interval for strictly uniform timestamps."""
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("Timestamp tolerance must be non-negative and finite.")
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("At least two one-dimensional timestamps are required.")
    if not np.isfinite(times).all():
        raise ValueError("Timestamps contain missing or non-finite values.")

    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise ValueError("Timestamps must be strictly increasing.")

    median_interval = float(np.median(intervals))
    if not np.allclose(
        intervals,
        median_interval,
        rtol=relative_tolerance,
        atol=1e-9,
    ):
        raise ValueError(
            "Timestamps are not uniformly sampled within the configured "
            f"{relative_tolerance:g} relative tolerance; resample explicitly "
            "before filtering."
        )
    return 1.0 / median_interval, median_interval


def _sosfiltfilt_padlen(sos: np.ndarray) -> int:
    zeros_at_origin = min(
        int(np.count_nonzero(sos[:, 2] == 0)),
        int(np.count_nonzero(sos[:, 5] == 0)),
    )
    return 3 * (2 * len(sos) + 1 - zeros_at_origin)


def _interpolate_short_gaps(
    values: np.ndarray,
    valid_mask: np.ndarray,
    max_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    filled = np.asarray(values, dtype=float).copy()
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(filled)
    filled[~valid] = np.nan
    interpolated = np.zeros(filled.shape, dtype=bool)

    if max_gap_samples < 1:
        return filled, interpolated

    for start, end in _finite_runs(~np.isfinite(filled)):
        gap_length = end - start
        if (
            gap_length <= max_gap_samples
            and start > 0
            and end < filled.size
            and np.isfinite(filled[start - 1])
            and np.isfinite(filled[end])
        ):
            filled[start:end] = np.linspace(
                filled[start - 1],
                filled[end],
                gap_length + 2,
            )[1:-1]
            interpolated[start:end] = True
    return filled, interpolated


def preprocess_channel(
    values: np.ndarray,
    valid_mask: np.ndarray,
    sampling_rate_hz: float,
    settings: PreprocessingSettings = DEFAULT_SETTINGS,
) -> ChannelStages:
    """Fill only short internal gaps, then process finite segments independently."""
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("Sampling rate must be a positive finite value.")
    if not 0 < settings.low_cut_hz < settings.high_cut_hz:
        raise ValueError("Bandpass frequencies must satisfy 0 < low < high.")
    if settings.high_cut_hz >= sampling_rate_hz / 2:
        raise ValueError(
            "The high cutoff must be below the Nyquist frequency "
            f"({sampling_rate_hz / 2:g} Hz)."
        )
    if settings.filter_order < 1:
        raise ValueError("Filter order must be a positive integer.")
    if (
        not np.isfinite(settings.max_interpolation_gap_seconds)
        or settings.max_interpolation_gap_seconds < 0
    ):
        raise ValueError("Maximum interpolation gap must be non-negative and finite.")
    if not 0 <= settings.min_skin_ratio <= 1:
        raise ValueError("Minimum skin ratio must be between zero and one.")
    if settings.detrend_type not in {"linear", "constant"}:
        raise ValueError("Detrend type must be 'linear' or 'constant'.")
    if settings.normalization != "zscore":
        raise ValueError("Only z-score normalization is currently supported.")

    source = np.asarray(values, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if source.ndim != 1 or valid.ndim != 1 or source.shape != valid.shape:
        raise ValueError("Values and validity mask must be matching 1D arrays.")

    max_gap_samples = int(
        np.floor(settings.max_interpolation_gap_seconds * sampling_rate_hz + 1e-9)
    )
    filled, interpolated_mask = _interpolate_short_gaps(
        source, valid, max_gap_samples
    )
    detrended = np.full(filled.shape, np.nan, dtype=float)
    filtered = np.full(filled.shape, np.nan, dtype=float)
    normalized = np.full(filled.shape, np.nan, dtype=float)

    sos = signal.butter(
        settings.filter_order,
        (settings.low_cut_hz, settings.high_cut_hz),
        btype="bandpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    minimum_segment_length = _sosfiltfilt_padlen(sos) + 1
    filtered_segment_count = 0
    short_segment_count = 0
    zero_variance_segment_count = 0

    for start, end in _finite_runs(np.isfinite(filled)):
        segment = filled[start:end]
        if segment.size < minimum_segment_length:
            short_segment_count += 1
            continue

        detrended_segment = signal.detrend(
            segment, type=settings.detrend_type
        )
        filtered_segment = signal.sosfiltfilt(sos, detrended_segment)
        detrended[start:end] = detrended_segment
        filtered[start:end] = filtered_segment
        filtered_segment_count += 1

        standard_deviation = float(np.std(filtered_segment))
        if not np.isfinite(standard_deviation) or standard_deviation <= np.finfo(float).eps:
            zero_variance_segment_count += 1
            continue
        normalized[start:end] = (
            filtered_segment - float(np.mean(filtered_segment))
        ) / standard_deviation

    return ChannelStages(
        filled=filled,
        detrended=detrended,
        filtered=filtered,
        normalized=normalized,
        interpolated_mask=interpolated_mask,
        filtered_segment_count=filtered_segment_count,
        short_segment_count=short_segment_count,
        zero_variance_segment_count=zero_variance_segment_count,
    )


def _statistics(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "finite_count": int(finite.size),
        "missing_count": int(np.asarray(values).size - finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "standard_deviation": float(np.std(finite)) if finite.size else None,
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def _roi_validity(df: pd.DataFrame, roi: str, settings: PreprocessingSettings) -> tuple[
    np.ndarray, dict[str, int]
]:
    face_values = pd.to_numeric(
        df["face_detected"], errors="coerce"
    ).to_numpy(dtype=float)
    face_detected = np.isfinite(face_values) & (face_values > 0)
    skin_ratio = pd.to_numeric(
        df[f"{roi}_skin_ratio"], errors="coerce"
    ).to_numpy(dtype=float)
    skin_accepted = np.isfinite(skin_ratio) & (
        skin_ratio >= settings.min_skin_ratio
    )
    channels_finite = np.logical_and.reduce(
        [
            np.isfinite(
                pd.to_numeric(df[f"{roi}_{channel}"], errors="coerce").to_numpy(
                    dtype=float
                )
            )
            for channel in CHANNELS
        ]
    )
    valid = face_detected & skin_accepted & channels_finite
    return valid, {
        "face_not_detected": int((~face_detected).sum()),
        "skin_ratio_missing_nonfinite_or_below_threshold": int(
            (~skin_accepted).sum()
        ),
        "one_or_more_rgb_channels_nonfinite": int((~channels_finite).sum()),
        "unique_invalid_samples": int((~valid).sum()),
    }


def _plot_roi(
    plot_dir: Path,
    roi: str,
    times: np.ndarray,
    df: pd.DataFrame,
    stages: dict[str, ChannelStages],
    valid_mask: np.ndarray,
) -> None:
    colors = {"r": "tab:red", "g": "tab:green", "b": "tab:blue"}
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for channel in CHANNELS:
        raw = pd.to_numeric(
            df[f"{roi}_{channel}"], errors="coerce"
        ).to_numpy(dtype=float)
        axes[0].plot(
            times, raw, color=colors[channel], linewidth=0.7, label=channel.upper()
        )
        axes[1].plot(
            times,
            stages[channel].filtered,
            color=colors[channel],
            linewidth=0.7,
            label=channel.upper(),
        )
        axes[2].plot(
            times,
            stages[channel].normalized,
            color=colors[channel],
            linewidth=0.7,
            label=channel.upper(),
        )

    for start, end in _finite_runs(~valid_mask):
        left = times[start]
        right = times[min(end, len(times) - 1)]
        for axis in axes:
            axis.axvspan(left, right, color="gray", alpha=0.15)
    for start, end in _finite_runs(stages[CHANNELS[0]].interpolated_mask):
        left = times[start]
        right = times[min(end, len(times) - 1)]
        for axis in axes:
            axis.axvspan(left, right, color="orange", alpha=0.18)

    axes[0].set_title(f"{roi.replace('_', ' ').title()} - Before and After")
    axes[0].set_ylabel("Raw intensity")
    axes[1].set_ylabel("Filtered intensity")
    axes[2].set_ylabel("Normalized (z-score)")
    axes[2].set_xlabel("Time (seconds)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{roi}_preprocessing.png", dpi=180)
    plt.close(fig)


def preprocess_dataframe(
    df: pd.DataFrame,
    settings: PreprocessingSettings = DEFAULT_SETTINGS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"video_time_seconds", "face_detected"}
    for roi in ROI_NAMES:
        required.update(f"{roi}_{channel}" for channel in CHANNELS)
        required.add(f"{roi}_skin_ratio")
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Cannot preprocess an empty RGB signal CSV.")

    timestamps = pd.to_numeric(
        df["video_time_seconds"], errors="coerce"
    ).to_numpy(dtype=float)
    sampling_rate_hz, median_interval = infer_sampling_rate(
        timestamps, settings.timestamp_relative_tolerance
    )
    if settings.high_cut_hz >= sampling_rate_hz / 2:
        raise ValueError(
            "The high cutoff must be below the Nyquist frequency "
            f"({sampling_rate_hz / 2:g} Hz)."
        )
    max_gap_samples = int(
        np.floor(settings.max_interpolation_gap_seconds * sampling_rate_hz + 1e-9)
    )

    output = df.copy()
    roi_reports: dict[str, Any] = {}
    for roi in ROI_NAMES:
        valid_mask, invalid_counts = _roi_validity(df, roi, settings)
        channel_stages: dict[str, ChannelStages] = {}
        channel_reports: dict[str, Any] = {}
        for channel in CHANNELS:
            values = pd.to_numeric(
                df[f"{roi}_{channel}"], errors="coerce"
            ).to_numpy(dtype=float)
            stages = preprocess_channel(values, valid_mask, sampling_rate_hz, settings)
            channel_stages[channel] = stages
            for stage_name in ("filled", "detrended", "filtered", "normalized"):
                output[f"{roi}_{channel}_{stage_name}"] = getattr(
                    stages, stage_name
                )
            channel_reports[channel] = {
                "raw": _statistics(values),
                "filled": _statistics(stages.filled),
                "detrended": _statistics(stages.detrended),
                "filtered": _statistics(stages.filtered),
                "normalized": _statistics(stages.normalized),
            }

        interpolated_mask = channel_stages[CHANNELS[0]].interpolated_mask
        output[f"{roi}_valid_sample"] = valid_mask
        output[f"{roi}_interpolated_sample"] = interpolated_mask
        output[f"{roi}_unfilled_invalid_sample"] = ~(
            valid_mask | interpolated_mask
        )
        roi_reports[roi] = {
            "original_valid_sample_count": int(valid_mask.sum()),
            "original_invalid_sample_count": int((~valid_mask).sum()),
            "invalid_sample_reasons": invalid_counts,
            "short_gap_interpolated_sample_count": int(interpolated_mask.sum()),
            "remaining_missing_sample_count": int(
                np.count_nonzero(~np.isfinite(channel_stages[CHANNELS[0]].filled))
            ),
            "contiguous_segments_filtered_per_channel": {
                channel: channel_stages[channel].filtered_segment_count
                for channel in CHANNELS
            },
            "segments_skipped_as_too_short_per_channel": {
                channel: channel_stages[channel].short_segment_count
                for channel in CHANNELS
            },
            "segments_without_finite_variance_per_channel": {
                channel: channel_stages[channel].zero_variance_segment_count
                for channel in CHANNELS
            },
            "channels": channel_reports,
        }

    report = {
        "source_row_count": int(len(df)),
        "sampling_rate_hz_from_timestamps": sampling_rate_hz,
        "median_frame_interval_seconds": median_interval,
        "timestamp_relative_uniformity_tolerance": (
            settings.timestamp_relative_tolerance
        ),
        "settings": {
            "bandpass_hz": [settings.low_cut_hz, settings.high_cut_hz],
            "butterworth_order": settings.filter_order,
            "filter_implementation": "second-order sections; scipy.signal.sosfiltfilt",
            "detrend_type": settings.detrend_type,
            "normalization": settings.normalization,
            "minimum_skin_ratio": settings.min_skin_ratio,
            "maximum_interpolated_internal_gap_seconds": (
                settings.max_interpolation_gap_seconds
            ),
            "maximum_interpolated_internal_gap_samples": max_gap_samples,
            "long_and_edge_gaps": "preserved as missing; not interpolated",
            "invalid_samples": (
                "excluded before interpolation when face is not detected, "
                "skin ratio is invalid/below threshold, or any ROI RGB channel "
                "is non-finite"
            ),
            "short_filter_segments": (
                "left missing when segment length is insufficient for sosfiltfilt"
            ),
            "source_csv_modified": False,
        },
        "rois": roi_reports,
    }
    return output, report


def run_preprocessing(subject: str) -> tuple[Path, Path, Path]:
    processed_dir = PROJECT_ROOT / "datasets" / "processed" / subject
    source_csv = processed_dir / "rgb_signals.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(f"RGB signal CSV not found: {source_csv}")

    df = pd.read_csv(source_csv)
    output_df, report = preprocess_dataframe(df)

    cleaned_dir = PROJECT_ROOT / "datasets" / "cleaned" / subject
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    output_csv = cleaned_dir / "rgb_signals_preprocessed.csv"
    output_df.to_csv(output_csv, index=False, float_format="%.17g")

    report_dir = PROJECT_ROOT / "outputs" / "reports" / subject
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "preprocessing_quality.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    plot_dir = PROJECT_ROOT / "outputs" / "plots" / subject / "preprocessing"
    plot_dir.mkdir(parents=True, exist_ok=True)
    times = pd.to_numeric(
        df["video_time_seconds"], errors="coerce"
    ).to_numpy(dtype=float)
    for roi in ROI_NAMES:
        channel_stages = {
            channel: ChannelStages(
                filled=output_df[f"{roi}_{channel}_filled"].to_numpy(dtype=float),
                detrended=output_df[f"{roi}_{channel}_detrended"].to_numpy(
                    dtype=float
                ),
                filtered=output_df[f"{roi}_{channel}_filtered"].to_numpy(
                    dtype=float
                ),
                normalized=output_df[f"{roi}_{channel}_normalized"].to_numpy(
                    dtype=float
                ),
                interpolated_mask=output_df[
                    f"{roi}_interpolated_sample"
                ].to_numpy(dtype=bool),
                filtered_segment_count=0,
                short_segment_count=0,
                zero_variance_segment_count=0,
            )
            for channel in CHANNELS
        }
        _plot_roi(
            plot_dir,
            roi,
            times,
            df,
            channel_stages,
            output_df[f"{roi}_valid_sample"].to_numpy(dtype=bool),
        )

    LOGGER.info("Saved preprocessed signals: %s", output_csv)
    LOGGER.info("Saved preprocessing report: %s", report_path)
    LOGGER.info("Saved before/after plots: %s", plot_dir)
    return output_csv, report_path, plot_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ROI RGB signals without changing the source CSV."
    )
    parser.add_argument("--subject", default="subject3")
    args = parser.parse_args()
    if Path(args.subject).name != args.subject or args.subject in {".", ".."}:
        parser.error("--subject must be a single folder name, such as subject3")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_csv, report_path, plot_dir = run_preprocessing(args.subject)
    print(f"Preprocessed CSV: {output_csv}")
    print(f"Quality report: {report_path}")
    print(f"Before/after plots: {plot_dir}")


if __name__ == "__main__":
    main()
