import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROI_NAMES = ("forehead", "right_cheek", "left_cheek")
CHANNELS = ("r", "g", "b")
SKIN_RATIO_THRESHOLD = 0.20
OUTLIER_MAD_THRESHOLD = 6.0
DISTURBANCE_WINDOWS = {
    "around_1_5_seconds": (1.0, 2.0),
    "around_27_to_28_seconds": (27.0, 28.0),
}

LOGGER = logging.getLogger("signal_quality")


def _describe(values: pd.Series | np.ndarray) -> dict[str, Any]:
    numeric = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return {
        "sample_count": int(numeric.size),
        "finite_count": int(finite.size),
        "missing_count": int(np.isnan(numeric).sum()),
        "infinite_count": int(np.isinf(numeric).sum()),
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


def _robust_outlier_mask(values: np.ndarray) -> np.ndarray:
    mask = np.zeros(values.shape, dtype=bool)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return mask

    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)))
    if mad == 0:
        mask[finite_mask] = finite_values != median
    else:
        robust_z = np.abs(finite_values - median) / (1.4826 * mad)
        mask[finite_mask] = robust_z > OUTLIER_MAD_THRESHOLD
    return mask


def _frame_indices(mask: np.ndarray, frame_values: np.ndarray) -> list[int]:
    return [
        int(frame_values[index]) if np.isfinite(frame_values[index]) else int(index)
        for index in np.flatnonzero(mask)
    ]


def _temporal_report(
    df: pd.DataFrame, metadata: dict[str, Any] | None
) -> dict[str, Any]:
    times = pd.to_numeric(df["video_time_seconds"], errors="coerce").to_numpy(
        dtype=float
    )
    frames = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
    finite_times = times[np.isfinite(times)]
    time_diffs = np.diff(times)
    finite_intervals = time_diffs[np.isfinite(time_diffs)]
    positive_intervals = finite_intervals[finite_intervals > 0]
    median_interval = (
        float(np.median(positive_intervals)) if positive_intervals.size else None
    )
    expected_fps = None
    if metadata is not None:
        candidate_fps = metadata.get("fps")
        if isinstance(candidate_fps, (int, float)) and np.isfinite(candidate_fps):
            expected_fps = float(candidate_fps) if candidate_fps > 0 else None
    expected_interval = 1.0 / expected_fps if expected_fps else None
    frame_diffs = np.diff(frames)
    finite_frame_diffs = frame_diffs[np.isfinite(frame_diffs)]

    return {
        "row_count": int(len(df)),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "missing_or_invalid_frames": int(np.count_nonzero(~np.isfinite(frames))),
        "duplicate_frame_values": int(
            pd.Series(frames[np.isfinite(frames)]).duplicated().sum()
        ),
        "non_increasing_frame_steps": int(
            np.count_nonzero(finite_frame_diffs <= 0)
        ),
        "frame_gaps": int(np.count_nonzero(finite_frame_diffs > 1)),
        "missing_or_invalid_timestamps": int(np.count_nonzero(~np.isfinite(times))),
        "duplicate_timestamps": int(np.count_nonzero(time_diffs == 0)),
        "decreasing_timestamps": int(np.count_nonzero(time_diffs < 0)),
        "median_positive_interval_seconds": median_interval,
        "expected_fps_from_video_metadata": expected_fps,
        "expected_interval_from_video_metadata_seconds": expected_interval,
        "intervals_over_1_5x_median": (
            int(np.count_nonzero(positive_intervals > 1.5 * median_interval))
            if median_interval is not None
            else 0
        ),
        "max_absolute_interval_deviation_from_median_seconds": (
            float(np.max(np.abs(positive_intervals - median_interval)))
            if positive_intervals.size and median_interval is not None
            else None
        ),
        "first_timestamp_seconds": (
            float(finite_times[0]) if finite_times.size else None
        ),
        "last_timestamp_seconds": (
            float(finite_times[-1]) if finite_times.size else None
        ),
    }


def _window_diagnostics(
    df: pd.DataFrame,
    roi: str,
    times: np.ndarray,
    channel_values: dict[str, np.ndarray],
    derivative_values: dict[str, np.ndarray],
    derivative_times: np.ndarray,
    derivative_outliers: dict[str, np.ndarray],
    face_detected: np.ndarray,
    face_confidence: np.ndarray,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    skin_ratio = pd.to_numeric(
        df[f"{roi}_skin_ratio"], errors="coerce"
    ).to_numpy(dtype=float)

    for label, (start, end) in DISTURBANCE_WINDOWS.items():
        sample_mask = np.isfinite(times) & (times >= start) & (times <= end)
        before_mask = (
            np.isfinite(times) & (times >= start - 1.0) & (times < start)
        )
        after_mask = (
            np.isfinite(times) & (times > end) & (times <= end + 1.0)
        )
        baseline_mask = before_mask | after_mask
        derivative_mask = (
            np.isfinite(derivative_times)
            & (derivative_times >= start)
            & (derivative_times <= end)
        )
        window: dict[str, Any] = {
            "window_seconds": [start, end],
            "sample_count": int(np.count_nonzero(sample_mask)),
            "face_detection_rate": (
                float(np.mean(face_detected[sample_mask]))
                if np.any(sample_mask)
                else None
            ),
            "face_confidence_mean": (
                float(np.nanmean(face_confidence[sample_mask]))
                if np.any(sample_mask) and np.isfinite(face_confidence[sample_mask]).any()
                else None
            ),
            "skin_ratio_mean": (
                float(np.nanmean(skin_ratio[sample_mask]))
                if np.any(sample_mask) and np.isfinite(skin_ratio[sample_mask]).any()
                else None
            ),
            "skin_ratio_minimum": (
                float(np.nanmin(skin_ratio[sample_mask]))
                if np.any(sample_mask) and np.isfinite(skin_ratio[sample_mask]).any()
                else None
            ),
            "adjacent_context": {
                "before_seconds": [start - 1.0, start],
                "after_seconds": [end, end + 1.0],
                "face_detection_rate": (
                    float(np.mean(face_detected[baseline_mask]))
                    if np.any(baseline_mask)
                    else None
                ),
                "face_confidence_mean": (
                    float(np.nanmean(face_confidence[baseline_mask]))
                    if np.any(baseline_mask)
                    and np.isfinite(face_confidence[baseline_mask]).any()
                    else None
                ),
                "skin_ratio_mean": (
                    float(np.nanmean(skin_ratio[baseline_mask]))
                    if np.any(baseline_mask)
                    and np.isfinite(skin_ratio[baseline_mask]).any()
                    else None
                ),
            },
            "channel_diagnostics": {},
            "interpretation_note": (
                "Use channel co-movement, skin ratio, face confidence, and outlier "
                "timing as diagnostic evidence only. RGB measurements alone cannot "
                "distinguish motion, illumination, ROI instability, and genuine "
                "physiological variation conclusively."
            ),
        }
        for channel in CHANNELS:
            values = channel_values[channel][sample_mask]
            finite = values[np.isfinite(values)]
            baseline_values = channel_values[channel][baseline_mask]
            finite_baseline = baseline_values[np.isfinite(baseline_values)]
            diffs = derivative_values[channel][derivative_mask]
            finite_diffs = diffs[np.isfinite(diffs)]
            peak_index = (
                int(np.nanargmax(np.abs(diffs))) if finite_diffs.size else None
            )
            window_derivative_indices = np.flatnonzero(derivative_mask)
            window["channel_diagnostics"][channel] = {
                "mean": float(np.mean(finite)) if finite.size else None,
                "standard_deviation": (
                    float(np.std(finite, ddof=1))
                    if finite.size > 1
                    else (0.0 if finite.size == 1 else None)
                ),
                "adjacent_baseline_mean": (
                    float(np.mean(finite_baseline)) if finite_baseline.size else None
                ),
                "mean_difference_from_adjacent_baseline": (
                    float(np.mean(finite) - np.mean(finite_baseline))
                    if finite.size and finite_baseline.size
                    else None
                ),
                "mean_absolute_change_per_second": (
                    float(np.mean(np.abs(finite_diffs)))
                    if finite_diffs.size
                    else None
                ),
                "maximum_absolute_change_per_second": (
                    float(np.max(np.abs(finite_diffs)))
                    if finite_diffs.size
                    else None
                ),
                "maximum_change_time_seconds": (
                    float(derivative_times[window_derivative_indices[peak_index]])
                    if peak_index is not None
                    else None
                ),
                "sudden_change_count": int(
                    np.count_nonzero(derivative_outliers[channel] & derivative_mask)
                ),
            }
        reports[label] = window
    return reports


def _plot_roi(
    output_dir: Path,
    roi: str,
    times: np.ndarray,
    channel_values: dict[str, np.ndarray],
    derivative_times: np.ndarray,
    derivative_values: dict[str, np.ndarray],
    level_outliers: dict[str, np.ndarray],
    derivative_outliers: dict[str, np.ndarray],
) -> None:
    colors = {"r": "tab:red", "g": "tab:green", "b": "tab:blue"}

    fig, axis = plt.subplots(figsize=(14, 5))
    for channel in CHANNELS:
        axis.plot(
            times,
            channel_values[channel],
            color=colors[channel],
            linewidth=0.8,
            label=channel.upper(),
        )
    axis.set(title=f"{roi.replace('_', ' ').title()} - Raw RGB", xlabel="Time (s)")
    axis.set_ylabel("Mean pixel intensity")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{roi}_raw_rgb.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 5))
    for channel in CHANNELS:
        axis.plot(
            derivative_times,
            derivative_values[channel],
            color=colors[channel],
            linewidth=0.8,
            label=channel.upper(),
        )
    axis.set(
        title=f"{roi.replace('_', ' ').title()} - RGB Signal Derivatives",
        xlabel="Time (s)",
    )
    axis.set_ylabel("Intensity change per second")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{roi}_derivatives.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 5))
    for channel in CHANNELS:
        axis.plot(
            times,
            channel_values[channel],
            color=colors[channel],
            linewidth=0.7,
            label=f"{channel.upper()} signal",
        )
        mask = level_outliers[channel]
        axis.scatter(
            times[mask],
            channel_values[channel][mask],
            color=colors[channel],
            marker="x",
            s=30,
            label=f"{channel.upper()} level outlier",
        )
        change_mask = derivative_outliers[channel]
        axis.scatter(
            derivative_times[change_mask],
            channel_values[channel][1:][change_mask],
            color=colors[channel],
            marker="o",
            facecolors="none",
            s=50,
            label=f"{channel.upper()} sudden change",
        )
    axis.set(
        title=f"{roi.replace('_', ' ').title()} - Outlier Locations",
        xlabel="Time (s)",
    )
    axis.set_ylabel("Mean pixel intensity")
    axis.grid(alpha=0.3)
    axis.legend(ncol=2, fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / f"{roi}_outliers.png", dpi=180)
    plt.close(fig)


def _plot_comparison(output_dir: Path, reports: dict[str, dict[str, Any]]) -> None:
    labels = [roi.replace("_", " ").title() for roi in ROI_NAMES]
    valid_rates = [
        reports[roi]["roi_valid_sample_rate_percent"] for roi in ROI_NAMES
    ]
    mean_skin_ratios = [
        reports[roi]["skin_mask"]["ratio_mean_percent"] for roi in ROI_NAMES
    ]
    outlier_rates = [
        reports[roi]["outlier_summary"]["sudden_change_rate_percent"]
        for roi in ROI_NAMES
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = (
        ("Valid RGB samples (%)", valid_rates),
        ("Mean skin-mask ratio (%)", mean_skin_ratios),
        ("Sudden-change flags (%)", outlier_rates),
    )
    for axis, (title, values) in zip(axes, metrics):
        axis.bar(labels, values, color=("tab:blue", "tab:orange", "tab:green"))
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("ROI Signal Quality Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "roi_quality_comparison.png", dpi=180)
    plt.close(fig)


def inspect_signal_quality(
    subject: str,
    csv_path: Path,
    report_dir: Path,
    plot_dir: Path,
    metadata_path: Path,
) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(csv_path)
    required_columns = {
        "frame",
        "video_time_seconds",
        "face_detected",
        "face_confidence",
    }
    for roi in ROI_NAMES:
        required_columns.update(f"{roi}_{channel}" for channel in CHANNELS)
        required_columns.add(f"{roi}_skin_ratio")
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError(f"CSV contains no samples: {csv_path}")

    metadata = None
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
    else:
        LOGGER.warning("Video metadata not found; FPS-based cadence checks are omitted.")

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    times = pd.to_numeric(df["video_time_seconds"], errors="coerce").to_numpy(
        dtype=float
    )
    frames = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
    face_detected = (
        pd.to_numeric(df["face_detected"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
        > 0
    )
    face_confidence = pd.to_numeric(
        df["face_confidence"], errors="coerce"
    ).to_numpy(dtype=float)
    temporal = _temporal_report(df, metadata)
    face_report = {
        "detected_frames": int(face_detected.sum()),
        "detection_rate_percent": float(100 * face_detected.mean()),
        "confidence": _describe(face_confidence),
    }
    integrity_columns = (
        "frame",
        "video_time_seconds",
        "face_detected",
        "face_confidence",
        *(
            f"{roi}_{field}"
            for roi in ROI_NAMES
            for field in (*CHANNELS, "skin_ratio")
        ),
    )
    column_integrity = {
        column: _describe(df[column]) for column in integrity_columns
    }
    reports: dict[str, dict[str, Any]] = {}

    for roi in ROI_NAMES:
        channel_values = {
            channel: pd.to_numeric(
                df[f"{roi}_{channel}"], errors="coerce"
            ).to_numpy(dtype=float)
            for channel in CHANNELS
        }
        valid_rgb = np.logical_and.reduce(
            [np.isfinite(channel_values[channel]) for channel in CHANNELS]
        )
        skin_ratio = pd.to_numeric(
            df[f"{roi}_skin_ratio"], errors="coerce"
        ).to_numpy(dtype=float)
        ratio_accepted = np.isfinite(skin_ratio) & (
            skin_ratio >= SKIN_RATIO_THRESHOLD
        )

        derivative_times = times[1:]
        time_steps = np.diff(times)
        valid_steps = (
            np.isfinite(time_steps)
            & (time_steps > 0)
            & np.isfinite(times[:-1])
            & np.isfinite(times[1:])
        )
        derivative_times = np.where(valid_steps, derivative_times, np.nan)
        derivative_values: dict[str, np.ndarray] = {}
        level_outliers: dict[str, np.ndarray] = {}
        derivative_outliers: dict[str, np.ndarray] = {}
        channel_reports: dict[str, Any] = {}
        sudden_change_total = 0
        level_outlier_total = 0

        for channel in CHANNELS:
            values = channel_values[channel]
            diffs = np.full(max(len(df) - 1, 0), np.nan, dtype=float)
            frame_differences = np.full(max(len(df) - 1, 0), np.nan, dtype=float)
            if len(df) > 1:
                delta = np.diff(values)
                usable = valid_steps & np.isfinite(values[:-1]) & np.isfinite(values[1:])
                diffs[usable] = delta[usable] / time_steps[usable]
                frame_delta = np.diff(values)
                finite_pairs = np.isfinite(values[:-1]) & np.isfinite(values[1:])
                frame_differences[finite_pairs] = frame_delta[finite_pairs]
            derivative_values[channel] = diffs
            level_outliers[channel] = _robust_outlier_mask(values)
            derivative_outliers[channel] = _robust_outlier_mask(diffs)
            level_indices = _frame_indices(level_outliers[channel], frames)
            change_indices = _frame_indices(
                np.r_[False, derivative_outliers[channel]], frames
            )
            level_outlier_total += len(level_indices)
            sudden_change_total += len(change_indices)
            channel_reports[channel] = {
                "statistics": _describe(values),
                "frame_to_frame_variation": {
                    "mean_absolute_change": (
                        float(np.mean(np.abs(frame_differences[
                            np.isfinite(frame_differences)
                        ])))
                        if np.isfinite(frame_differences).any()
                        else None
                    ),
                    "mean_absolute_change_per_second": (
                        float(np.nanmean(np.abs(diffs)))
                        if np.isfinite(diffs).any()
                        else None
                    ),
                    "maximum_absolute_change_per_second": (
                        float(np.nanmax(np.abs(diffs)))
                        if np.isfinite(diffs).any()
                        else None
                    ),
                },
                "level_outliers": {
                    "method": "median absolute deviation; threshold > 6 robust standard deviations",
                    "count": len(level_indices),
                    "frame_indices": level_indices,
                },
                "sudden_changes": {
                    "method": "median absolute deviation on time-normalized first differences; threshold > 6 robust standard deviations",
                    "count": len(change_indices),
                    "frame_indices": change_indices,
                },
            }

        ratio_finite = skin_ratio[np.isfinite(skin_ratio)]
        reports[roi] = {
            "subject": subject,
            "roi": roi,
            "csv_path": str(csv_path),
            "sample_count": int(len(df)),
            "roi_valid_sample_count": int(valid_rgb.sum()),
            "roi_valid_sample_rate_percent": float(100 * valid_rgb.mean()),
            "missing_rgb_samples": int((~valid_rgb).sum()),
            "column_integrity": {
                column: column_integrity[column]
                for column in (
                    f"{roi}_{channel}" for channel in CHANNELS
                )
            }
            | {f"{roi}_skin_ratio": column_integrity[f"{roi}_skin_ratio"]},
            "face_detection": face_report,
            "skin_mask": {
                "ratio_threshold": SKIN_RATIO_THRESHOLD,
                "ratio_mean_percent": (
                    float(100 * np.mean(ratio_finite)) if ratio_finite.size else None
                ),
                "ratio_median_percent": (
                    float(100 * np.median(ratio_finite)) if ratio_finite.size else None
                ),
                "ratio_minimum_percent": (
                    float(100 * np.min(ratio_finite)) if ratio_finite.size else None
                ),
                "ratio_maximum_percent": (
                    float(100 * np.max(ratio_finite)) if ratio_finite.size else None
                ),
                "ratio_at_or_above_threshold_count": int(ratio_accepted.sum()),
                "ratio_at_or_above_threshold_rate_percent": float(
                    100 * ratio_accepted.mean()
                ),
                "valid_rgb_sample_count": int(valid_rgb.sum()),
                "valid_rgb_samples_are_extractor_accepted": True,
                "limitation": (
                    "RGB CSV does not retain skin-pixel counts. Finite RGB values "
                    "indicate samples accepted by the extractor, but the per-frame "
                    "minimum-pixel condition cannot be independently recomputed."
                ),
            },
            "channels": channel_reports,
            "outlier_summary": {
                "level_outlier_count_across_channels": level_outlier_total,
                "sudden_change_count_across_channels": sudden_change_total,
                "sudden_change_rate_percent": float(
                    100 * sudden_change_total / max(len(df) * len(CHANNELS), 1)
                ),
                "not_removed": True,
            },
            "temporal_consistency": temporal,
            "disturbance_windows": _window_diagnostics(
                df,
                roi,
                times,
                channel_values,
                derivative_values,
                derivative_times,
                derivative_outliers,
                face_detected,
                face_confidence,
            ),
        }
        _plot_roi(
            plot_dir,
            roi,
            times,
            channel_values,
            derivative_times,
            derivative_values,
            level_outliers,
            derivative_outliers,
        )
        output_path = report_dir / f"{roi}_quality.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(reports[roi], file, indent=2, allow_nan=False)
        LOGGER.info("Saved ROI report: %s", output_path)

    _plot_comparison(plot_dir, reports)
    summary = {
        "subject": subject,
        "source_csv": str(csv_path),
        "sample_count": int(len(df)),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "face_detection": face_report,
        "column_integrity": column_integrity,
        "temporal_consistency": temporal,
        "disturbance_windows_seconds": DISTURBANCE_WINDOWS,
        "roi_summary": {
            roi: {
                "valid_rgb_sample_count": reports[roi]["roi_valid_sample_count"],
                "valid_rgb_sample_rate_percent": reports[roi][
                    "roi_valid_sample_rate_percent"
                ],
                "mean_skin_ratio_percent": reports[roi]["skin_mask"][
                    "ratio_mean_percent"
                ],
                "sudden_change_count": reports[roi]["outlier_summary"][
                    "sudden_change_count_across_channels"
                ],
            }
            for roi in ROI_NAMES
        },
        "interpretation_limit": (
            "Outlier flags identify observations for review and are not grounds for "
            "automatic removal. RGB-only diagnostics cannot conclusively attribute "
            "disturbances to motion, illumination, ROI instability, or physiology."
        ),
    }
    summary_path = report_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    LOGGER.info("Saved summary report: %s", summary_path)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect extracted per-ROI RGB signal quality without modifying samples."
    )
    parser.add_argument("--subject", default="subject3")
    args = parser.parse_args()
    if Path(args.subject).name != args.subject or args.subject in {".", ".."}:
        parser.error("--subject must be a single folder name, such as subject3")

    processed_dir = PROJECT_ROOT / "datasets" / "processed"
    csv_path = processed_dir / args.subject / "rgb_signals.csv"
    metadata_path = processed_dir / f"{args.subject}_video_metadata.json"
    report_dir = PROJECT_ROOT / "outputs" / "reports" / args.subject
    plot_dir = PROJECT_ROOT / "outputs" / "plots" / args.subject / "quality"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not csv_path.is_file():
        raise FileNotFoundError(f"RGB signal CSV not found: {csv_path}")
    inspect_signal_quality(
        args.subject, csv_path, report_dir, plot_dir, metadata_path
    )
    print(f"Quality reports saved to: {report_dir}")
    print(f"Quality plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
