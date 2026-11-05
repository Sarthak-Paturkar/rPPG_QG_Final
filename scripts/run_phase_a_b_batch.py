import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
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


DEFAULT_SUBJECTS = (
    "subject3",
    "subject8",
    "subject11",
    "subject12",
    "subject13",
    "subject15",
    "subject16",
    "subject20",
    "subject22",
    "subject23",
    "subject24",
    "subject25",
    "subject26",
    "subject27",
    "subject30",
    "subject32",
    "subject33",
    "subject35",
    "subject36",
    "subject38",
    "subject39",
    "subject40",
    "subject41",
    "subject42",
    "subject43",
    "subject44",
    "subject45",
    "subject47",
)
ROI_NAMES = ("forehead", "right_cheek", "left_cheek")
CHANNELS = ("r", "g", "b")
RGB_COLUMNS = tuple(
    column
    for roi in ROI_NAMES
    for column in (
        *(f"{roi}_{channel}" for channel in CHANNELS),
        f"{roi}_skin_ratio",
    )
)
RGB_REQUIRED_COLUMNS = {
    "frame",
    "video_time_seconds",
    "face_detected",
    "face_confidence",
    *RGB_COLUMNS,
}
STAGE_NAMES = ("rgb_extraction", "phase_a", "phase_b")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
LOGGER = logging.getLogger("phase_a_b_batch")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _subject_paths(subject: str) -> dict[str, Path]:
    raw_dir = PROJECT_ROOT / "datasets" / "raw" / subject
    rgb_csv = PROJECT_ROOT / "datasets" / "processed" / subject / "rgb_signals.csv"
    report_dir = PROJECT_ROOT / "outputs" / "reports" / subject
    quality_plot_dir = PROJECT_ROOT / "outputs" / "plots" / subject / "quality"
    cleaned_csv = (
        PROJECT_ROOT
        / "datasets"
        / "cleaned"
        / subject
        / "rgb_signals_preprocessed.csv"
    )
    preprocessing_plot_dir = (
        PROJECT_ROOT / "outputs" / "plots" / subject / "preprocessing"
    )
    return {
        "video": raw_dir / "vid.avi",
        "ground_truth": raw_dir / "ground_truth.txt",
        "rgb_csv": rgb_csv,
        "phase_a_summary": report_dir / "summary.json",
        "phase_a_report_dir": report_dir,
        "phase_a_plot_dir": quality_plot_dir,
        "phase_b_csv": cleaned_csv,
        "phase_b_report": report_dir / "preprocessing_quality.json",
        "phase_b_plot_dir": preprocessing_plot_dir,
    }


def _output_path_map(paths: dict[str, Path]) -> dict[str, list[str]]:
    return {
        "rgb_extraction": [str(paths["rgb_csv"])],
        "phase_a": [
            str(paths["phase_a_summary"]),
            *(
                str(paths["phase_a_report_dir"] / f"{roi}_quality.json")
                for roi in ROI_NAMES
            ),
            *(
                str(paths["phase_a_plot_dir"] / f"{roi}_{plot}.png")
                for roi in ROI_NAMES
                for plot in ("raw_rgb", "derivatives", "outliers")
            ),
            str(paths["phase_a_plot_dir"] / "roi_quality_comparison.png"),
        ],
        "phase_b": [
            str(paths["phase_b_csv"]),
            str(paths["phase_b_report"]),
            *(
                str(paths["phase_b_plot_dir"] / f"{roi}_preprocessing.png")
                for roi in ROI_NAMES
            ),
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _is_valid_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= len(PNG_SIGNATURE):
        return False
    with path.open("rb") as file:
        return file.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE


def _read_valid_rgb_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return None, f"RGB CSV is absent or empty: {path}"
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        return None, f"Could not parse RGB CSV {path}: {error}"

    missing_columns = sorted(RGB_REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        return None, f"RGB CSV is missing columns: {missing_columns}"
    if len(df) < 2:
        return None, f"RGB CSV has fewer than two rows: {path}"

    frames = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_numeric(
        df["video_time_seconds"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(frames).all() or not np.isfinite(timestamps).all():
        return None, "RGB CSV has non-finite frame numbers or timestamps."
    if not np.all(np.diff(frames) == 1):
        return None, "RGB CSV frame indices are not consecutive."
    if not np.all(np.diff(timestamps) > 0):
        return None, "RGB CSV timestamps are not strictly increasing."
    return df, None


def validate_rgb_output(path: Path) -> tuple[bool, str, pd.DataFrame | None]:
    df, error = _read_valid_rgb_csv(path)
    if error is not None or df is None:
        return False, error or f"Invalid RGB CSV: {path}", None
    valid_counts = {
        roi: int(
            np.logical_and.reduce(
                [
                    np.isfinite(
                        pd.to_numeric(df[f"{roi}_{channel}"], errors="coerce")
                    )
                    for channel in CHANNELS
                ]
            ).sum()
        )
        for roi in ROI_NAMES
    }
    return True, f"RGB CSV valid ({len(df)} frames; finite ROI RGB counts: {valid_counts})", df


def validate_phase_a_outputs(
    subject: str, paths: dict[str, Path], source_df: pd.DataFrame
) -> tuple[bool, str]:
    try:
        summary = _load_json(paths["phase_a_summary"])
        if summary.get("subject") != subject:
            return False, "Phase A summary subject does not match."
        if int(summary.get("sample_count", -1)) != len(source_df):
            return False, "Phase A summary sample count does not match the RGB CSV."
        source_csv_in_report = Path(str(summary.get("source_csv", "")))
        if not source_csv_in_report.is_file() or (
            source_csv_in_report.resolve() != paths["rgb_csv"].resolve()
        ):
            return False, "Phase A summary does not reference this subject's RGB CSV."

        roi_summary = summary.get("roi_summary")
        if not isinstance(roi_summary, dict):
            return False, "Phase A summary has no ROI summary."

        for roi in ROI_NAMES:
            report_path = paths["phase_a_report_dir"] / f"{roi}_quality.json"
            report = _load_json(report_path)
            if report.get("subject") != subject or report.get("roi") != roi:
                return False, f"Phase A ROI report is for the wrong subject/ROI: {report_path}"
            if int(report.get("sample_count", -1)) != len(source_df):
                return False, f"Phase A sample count mismatch in {report_path}"
            channels = report.get("channels")
            if not isinstance(channels, dict):
                return False, f"Phase A report has no channel diagnostics: {report_path}"
            expected_valid_count = int(
                np.logical_and.reduce(
                    [
                        np.isfinite(
                            pd.to_numeric(
                                source_df[f"{roi}_{channel}"], errors="coerce"
                            )
                        )
                        for channel in CHANNELS
                    ]
                ).sum()
            )
            roi_summary_entry = roi_summary.get(roi, {})
            if int(roi_summary_entry.get("valid_rgb_sample_count", -1)) != expected_valid_count:
                return False, f"Phase A summary valid-sample count does not match for {roi}."
            if int(report.get("roi_valid_sample_count", -1)) != expected_valid_count:
                return False, f"Phase A valid-sample count does not match for {roi}."
            for channel in CHANNELS:
                values = pd.to_numeric(
                    source_df[f"{roi}_{channel}"], errors="coerce"
                ).to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                stats = channels.get(channel, {}).get("statistics", {})
                expected_stats = {
                    "finite_count": int(finite.size),
                    "missing_count": int(values.size - finite.size),
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
                for key, expected in expected_stats.items():
                    actual = stats.get(key)
                    if expected is None:
                        if actual is not None:
                            return False, f"Phase A {roi} {channel} {key} mismatch."
                    elif actual is None or not math.isclose(
                        float(actual), expected, rel_tol=1e-8, abs_tol=1e-10
                    ):
                        return False, (
                            f"Phase A {roi} {channel} {key} does not match "
                            "the current RGB CSV."
                        )

        expected_plots = [
            *(
                paths["phase_a_plot_dir"] / f"{roi}_{plot}.png"
                for roi in ROI_NAMES
                for plot in ("raw_rgb", "derivatives", "outliers")
            ),
            paths["phase_a_plot_dir"] / "roi_quality_comparison.png",
        ]
        missing_plots = [str(path) for path in expected_plots if not _is_valid_png(path)]
        if missing_plots:
            return False, f"Phase A plots are absent or invalid: {missing_plots}"
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
        return False, f"Phase A output validation failed: {error}"
    return True, "Phase A reports and plots are valid and match the current RGB data."


def validate_phase_b_outputs(
    paths: dict[str, Path], source_df: pd.DataFrame
) -> tuple[bool, str]:
    try:
        report = _load_json(paths["phase_b_report"])
        if int(report.get("source_row_count", -1)) != len(source_df):
            return False, "Phase B report sample count does not match the RGB CSV."
        rate = float(report.get("sampling_rate_hz_from_timestamps", 0))
        if not np.isfinite(rate) or rate <= 0:
            return False, "Phase B report has an invalid timestamp-derived rate."
        settings = report.get("settings", {})
        if settings.get("bandpass_hz") != list(PREPROCESSING_BAND_HZ):
            return False, "Phase B report bandpass differs from configured settings."
        if settings.get("butterworth_order") != PREPROCESSING_FILTER_ORDER:
            return False, "Phase B report filter order differs from configured settings."
        if settings.get("minimum_skin_ratio") != PREPROCESSING_MIN_SKIN_RATIO:
            return False, "Phase B report skin threshold differs from configured settings."
        if (
            settings.get("maximum_interpolated_internal_gap_seconds")
            != PREPROCESSING_MAX_INTERPOLATION_GAP_SECONDS
        ):
            return False, "Phase B report interpolation limit differs from configured settings."
        if settings.get("detrend_type") != PREPROCESSING_DETREND_TYPE:
            return False, "Phase B report detrending differs from configured settings."
        if settings.get("normalization") != PREPROCESSING_NORMALIZATION:
            return False, "Phase B report normalization differs from configured settings."
        if (
            report.get("timestamp_relative_uniformity_tolerance")
            != PREPROCESSING_TIMESTAMP_RELATIVE_TOLERANCE
        ):
            return False, "Phase B timestamp tolerance differs from configured settings."
        if settings.get("source_csv_modified") is not False:
            return False, "Phase B report does not confirm the source CSV was preserved."
        if not isinstance(report.get("rois"), dict):
            return False, "Phase B report has no ROI results."

        if not paths["phase_b_csv"].is_file() or paths["phase_b_csv"].stat().st_size == 0:
            return False, f"Preprocessed CSV is absent or empty: {paths['phase_b_csv']}"
        processed_df = pd.read_csv(paths["phase_b_csv"])
        if len(processed_df) != len(source_df):
            return False, "Preprocessed CSV row count does not match the source."
        for column in source_df.columns:
            if column not in processed_df:
                return False, f"Preprocessed CSV is missing original column {column}."
            source_values = source_df[column]
            output_values = processed_df[column]
            if pd.api.types.is_numeric_dtype(source_values):
                if not np.allclose(
                    source_values.to_numpy(dtype=float),
                    output_values.to_numpy(dtype=float),
                    rtol=0,
                    atol=5e-13,
                    equal_nan=True,
                ):
                    return False, f"Preprocessed CSV altered source column {column}."
            elif not source_values.equals(output_values):
                return False, f"Preprocessed CSV altered source column {column}."

        for roi in ROI_NAMES:
            if roi not in report["rois"]:
                return False, f"Phase B report is missing ROI {roi}."
            required_columns = [
                f"{roi}_{channel}_{stage}"
                for channel in CHANNELS
                for stage in ("filled", "detrended", "filtered", "normalized")
            ]
            required_columns.extend(
                (
                    f"{roi}_valid_sample",
                    f"{roi}_interpolated_sample",
                    f"{roi}_unfilled_invalid_sample",
                )
            )
            missing = [column for column in required_columns if column not in processed_df]
            if missing:
                return False, f"Preprocessed CSV is missing columns: {missing}"

        expected_plots = [
            paths["phase_b_plot_dir"] / f"{roi}_preprocessing.png"
            for roi in ROI_NAMES
        ]
        missing_plots = [str(path) for path in expected_plots if not _is_valid_png(path)]
        if missing_plots:
            return False, f"Phase B plots are absent or invalid: {missing_plots}"
    except (
        OSError,
        json.JSONDecodeError,
        pd.errors.ParserError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        return False, f"Phase B output validation failed: {error}"
    return True, "Phase B CSV, report, and plots are valid; source columns are preserved."


def _validate_video(path: Path) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, f"Video is absent or empty: {path}"
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False, f"OpenCV could not open video: {path}"
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded, frame = capture.read()
        if not np.isfinite(fps) or fps <= 0 or frame_count < 1:
            return False, f"Video has invalid FPS/frame count: {path}"
        if not decoded or frame is None:
            return False, f"Video could not decode its first frame: {path}"
        return True, f"Video validated ({frame_count} frames at {fps:.6f} FPS)."
    finally:
        capture.release()


def _download_inputs(
    subject: str,
    paths: dict[str, Path],
    need_video: bool,
    rclone_executable: str,
) -> None:
    raw_parent = PROJECT_ROOT / "datasets" / "raw"
    raw_parent.mkdir(parents=True, exist_ok=True)
    raw_dir = paths["video"].parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    if paths["ground_truth"].exists():
        if not paths["ground_truth"].is_file() or paths["ground_truth"].stat().st_size == 0:
            raise RuntimeError(
                f"Existing ground-truth file is invalid; refusing to overwrite: "
                f"{paths['ground_truth']}"
            )
    if need_video and paths["video"].exists():
        valid, reason = _validate_video(paths["video"])
        if not valid:
            raise RuntimeError(
                f"Existing video is invalid; refusing to overwrite: {reason}"
            )

    missing_ground_truth = not paths["ground_truth"].exists()
    missing_video = need_video and not paths["video"].exists()
    if not missing_ground_truth and not missing_video:
        return

    with tempfile.TemporaryDirectory(
        prefix=f".{subject}_batch_", dir=raw_parent
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        staged_files: list[tuple[Path, Path, str]] = []
        requests = []
        if missing_ground_truth:
            requests.append(
                (
                    f"gdrive:UBFC_DATASET/DATASET_2/{subject}/ground_truth.txt",
                    staging_dir / "ground_truth.txt",
                    paths["ground_truth"],
                    "ground truth",
                )
            )
        if missing_video:
            requests.append(
                (
                    f"gdrive:UBFC_DATASET/DATASET_2/{subject}/vid.avi",
                    staging_dir / "vid.avi",
                    paths["video"],
                    "video",
                )
            )

        for remote, staged_path, local_path, label in requests:
            LOGGER.info("%s: downloading %s from %s", subject, label, remote)
            result = subprocess.run(
                [rclone_executable, "copyto", remote, str(staged_path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"rclone failed to download {label} (exit {result.returncode}): "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            if not staged_path.is_file() or staged_path.stat().st_size == 0:
                raise RuntimeError(f"rclone produced an absent or empty {label} file.")
            if label == "video":
                valid, reason = _validate_video(staged_path)
                if not valid:
                    raise RuntimeError(f"Downloaded video failed validation: {reason}")
            staged_files.append((staged_path, local_path, label))

        for staged_path, local_path, label in staged_files:
            if local_path.exists():
                raise RuntimeError(
                    f"Refusing to overwrite local {label} created during download: "
                    f"{local_path}"
                )
            os.replace(staged_path, local_path)
            LOGGER.info("%s: saved downloaded %s to %s", subject, label, local_path)


def _run_stage(command: list[str], subject: str, stage: str) -> None:
    LOGGER.info("%s: starting %s", subject, stage)
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout.strip():
        LOGGER.info("%s: %s output:\n%s", subject, stage, result.stdout.strip())
    if result.returncode != 0:
        tail = result.stderr.strip()[-4000:] or result.stdout.strip()[-4000:]
        raise RuntimeError(
            f"{stage} exited with code {result.returncode}. {tail}"
        )


def _new_stage_record(outputs: list[str]) -> dict[str, Any]:
    return {"status": "pending", "outputs": outputs, "message": None}


def _persist_summary(
    output_path: Path, subjects: list[str], records: dict[str, dict[str, Any]]
) -> None:
    ordered_records = [records[subject] for subject in subjects]
    completed = [
        record["subject"]
        for record in ordered_records
        if record["status"] == "completed"
    ]
    failed = [
        record["subject"]
        for record in ordered_records
        if record["status"] == "failed"
    ]
    skipped = [
        record["subject"]
        for record in ordered_records
        if record.get("skipped_because_outputs_already_valid", False)
    ]
    statuses = {record["status"] for record in ordered_records}
    if statuses.issubset({"completed", "failed"}):
        run_status = "completed_with_failures" if failed else "completed"
    elif "running" in statuses:
        run_status = "running"
    else:
        run_status = "pending"
    stage_statuses = {
        stage: {
            status: sum(
                record["stages"][stage]["status"] == status
                for record in ordered_records
            )
            for status in ("completed", "skipped_valid", "failed", "pending", "not_run")
        }
        for stage in STAGE_NAMES
    }
    summary = {
        "schema_version": 1,
        "status": run_status,
        "updated_at": _utc_now(),
        "total_subjects": len(subjects),
        "successfully_completed_count": len(completed),
        "successfully_completed_subjects": completed,
        "failed_count": len(failed),
        "failed_subjects": failed,
        "subjects_skipped_because_outputs_already_valid": skipped,
        "stage_status_counts": stage_statuses,
        "subjects": ordered_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(summary, temporary_file, indent=2, allow_nan=False)

        for attempt in range(5):
            try:
                os.replace(temporary_path, output_path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1 * (2**attempt))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _process_subject(
    subject: str,
    rclone_executable: str,
    record: dict[str, Any],
    subjects: list[str],
    records: dict[str, dict[str, Any]],
    summary_path: Path,
) -> None:
    paths = _subject_paths(subject)
    current_stage = "input_download"

    def update_summary() -> None:
        record["updated_at"] = _utc_now()
        _persist_summary(summary_path, subjects, records)

    try:
        rgb_valid, rgb_message, source_df = validate_rgb_output(paths["rgb_csv"])
        current_stage = "inputs"
        _download_inputs(
            subject,
            paths,
            need_video=not rgb_valid,
            rclone_executable=rclone_executable,
        )
        record["inputs"] = {
            "status": "ready",
            "ground_truth": str(paths["ground_truth"]),
            "video": str(paths["video"]),
            "video_needed_for_rgb_extraction": not rgb_valid,
        }
        update_summary()

        current_stage = "rgb_extraction"
        stage_record = record["stages"]["rgb_extraction"]
        if rgb_valid:
            stage_record.update(status="skipped_valid", message=rgb_message)
        else:
            if paths["rgb_csv"].exists():
                LOGGER.warning(
                    "%s: existing RGB CSV failed validation and will be regenerated: %s",
                    subject,
                    rgb_message,
                )
            stage_record.update(status="running", message=rgb_message)
            update_summary()
            _run_stage(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "src" / "roi_extraction" / "extract_rgb_signals.py"),
                    "--subject",
                    subject,
                ],
                subject,
                "RGB extraction",
            )
            rgb_valid, rgb_message, source_df = validate_rgb_output(paths["rgb_csv"])
            if not rgb_valid or source_df is None:
                raise RuntimeError(f"RGB output verification failed: {rgb_message}")
            stage_record.update(status="completed", message=rgb_message)
        update_summary()

        if source_df is None:
            source_df, rgb_error = _read_valid_rgb_csv(paths["rgb_csv"])
            if source_df is None:
                raise RuntimeError(f"RGB CSV became invalid: {rgb_error}")

        current_stage = "phase_a"
        phase_a_valid, phase_a_message = validate_phase_a_outputs(
            subject, paths, source_df
        )
        stage_record = record["stages"]["phase_a"]
        if phase_a_valid:
            stage_record.update(status="skipped_valid", message=phase_a_message)
        else:
            if any(
                (paths["phase_a_report_dir"] / f"{roi}_quality.json").exists()
                for roi in ROI_NAMES
            ) or paths["phase_a_summary"].exists():
                LOGGER.warning(
                    "%s: existing Phase A outputs failed validation and will be regenerated: %s",
                    subject,
                    phase_a_message,
                )
            stage_record.update(status="running", message=phase_a_message)
            update_summary()
            _run_stage(
                [
                    sys.executable,
                    "-m",
                    "src.validation.inspect_signal_quality",
                    "--subject",
                    subject,
                ],
                subject,
                "Phase A",
            )
            phase_a_valid, phase_a_message = validate_phase_a_outputs(
                subject, paths, source_df
            )
            if not phase_a_valid:
                raise RuntimeError(
                    f"Phase A output verification failed: {phase_a_message}"
                )
            stage_record.update(status="completed", message=phase_a_message)
        update_summary()

        current_stage = "phase_b"
        phase_b_valid, phase_b_message = validate_phase_b_outputs(paths, source_df)
        stage_record = record["stages"]["phase_b"]
        if phase_b_valid:
            stage_record.update(status="skipped_valid", message=phase_b_message)
        else:
            if paths["phase_b_csv"].exists() or paths["phase_b_report"].exists():
                LOGGER.warning(
                    "%s: existing Phase B outputs failed validation and will be regenerated: %s",
                    subject,
                    phase_b_message,
                )
            stage_record.update(status="running", message=phase_b_message)
            update_summary()
            _run_stage(
                [
                    sys.executable,
                    "-m",
                    "src.preprocessing.preprocess_signals",
                    "--subject",
                    subject,
                ],
                subject,
                "Phase B",
            )
            phase_b_valid, phase_b_message = validate_phase_b_outputs(
                paths, source_df
            )
            if not phase_b_valid:
                raise RuntimeError(
                    f"Phase B output verification failed: {phase_b_message}"
                )
            stage_record.update(status="completed", message=phase_b_message)
        update_summary()

        current_stage = "final_verification"
        rgb_valid, rgb_message, verified_df = validate_rgb_output(paths["rgb_csv"])
        if not rgb_valid or verified_df is None:
            raise RuntimeError(f"Final RGB verification failed: {rgb_message}")
        phase_a_valid, phase_a_message = validate_phase_a_outputs(
            subject, paths, verified_df
        )
        if not phase_a_valid:
            raise RuntimeError(f"Final Phase A verification failed: {phase_a_message}")
        phase_b_valid, phase_b_message = validate_phase_b_outputs(paths, verified_df)
        if not phase_b_valid:
            raise RuntimeError(f"Final Phase B verification failed: {phase_b_message}")

        current_stage = "video_cleanup"
        cleanup_status = "not_present"
        video_path = paths["video"]
        if video_path.exists():
            valid_video, video_message = _validate_video(video_path)
            if not valid_video:
                raise RuntimeError(
                    f"All outputs passed, but local video failed pre-cleanup "
                    f"validation and was retained: {video_message}"
                )
            video_path.unlink()
            cleanup_status = "deleted_after_output_verification"
            LOGGER.info("%s: deleted verified local video %s", subject, video_path)
        record["cleanup"] = {
            "video_status": cleanup_status,
            "video_path": str(video_path),
            "outputs_verified_before_deletion": True,
            "ground_truth_retained": paths["ground_truth"].is_file(),
        }
        record["status"] = "completed"
        record["completed_at"] = _utc_now()
        record["skipped_because_outputs_already_valid"] = all(
            record["stages"][stage]["status"] == "skipped_valid"
            for stage in STAGE_NAMES
        )
        LOGGER.info("%s: all requested outputs verified.", subject)
    except Exception as error:
        LOGGER.exception("%s: failed during %s", subject, current_stage)
        if current_stage == "inputs":
            record["inputs"] = {"status": "failed", "message": str(error)}
        for stage in STAGE_NAMES:
            stage_record = record["stages"][stage]
            if stage == current_stage:
                stage_record.update(status="failed", message=str(error))
            elif stage_record["status"] in {"pending", "running"}:
                stage_record.update(
                    status="not_run",
                    message=f"Not run because {current_stage} failed.",
                )
        record["errors"].append(
            {"stage": current_stage, "message": str(error), "at": _utc_now()}
        )
        record["status"] = "failed"
        record["completed_at"] = _utc_now()
    finally:
        update_summary()


def run_batch(subjects: tuple[str, ...] = DEFAULT_SUBJECTS) -> Path:
    if not subjects:
        raise ValueError("At least one subject must be supplied.")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Subject list contains duplicates.")
    invalid_subjects = [
        subject for subject in subjects if re.fullmatch(r"subject\d+", subject) is None
    ]
    if invalid_subjects:
        raise ValueError(f"Invalid subject folder names: {invalid_subjects}")

    summary_path = PROJECT_ROOT / "outputs" / "reports" / "batch_phase_a_b_summary.json"
    output_path_map = {
        subject: _output_path_map(_subject_paths(subject)) for subject in subjects
    }
    records: dict[str, dict[str, Any]] = {
        subject: {
            "subject": subject,
            "status": "pending",
            "started_at": _utc_now(),
            "completed_at": None,
            "updated_at": _utc_now(),
            "skipped_because_outputs_already_valid": False,
            "inputs": {"status": "pending"},
            "stages": {
                stage: _new_stage_record(output_path_map[subject][stage])
                for stage in STAGE_NAMES
            },
            "errors": [],
        }
        for subject in subjects
    }
    _persist_summary(summary_path, list(subjects), records)

    rclone_executable = shutil.which("rclone")
    if rclone_executable is None:
        for subject, record in records.items():
            record["status"] = "failed"
            record["completed_at"] = _utc_now()
            record["errors"].append(
                {
                    "stage": "input_download",
                    "message": "rclone executable was not found on PATH.",
                    "at": _utc_now(),
                }
            )
        _persist_summary(summary_path, list(subjects), records)
        return summary_path

    for subject in subjects:
        record = records[subject]
        record["status"] = "running"
        record["updated_at"] = _utc_now()
        _persist_summary(summary_path, list(subjects), records)
        LOGGER.info("========== %s (%d/%d) ==========", subject, subjects.index(subject) + 1, len(subjects))
        _process_subject(
            subject,
            rclone_executable,
            record,
            list(subjects),
            records,
            summary_path,
        )

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download, extract, inspect, and preprocess the validated UBFC-rPPG "
            "subject set. Existing valid phase outputs are skipped."
        )
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=list(DEFAULT_SUBJECTS),
        help="Optional subject subset; defaults to the 28 validated subjects.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    summary_path = run_batch(tuple(args.subjects))
    print(f"Batch summary: {summary_path}")
    summary = _load_json(summary_path)
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
