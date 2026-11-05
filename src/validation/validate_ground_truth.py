import numpy as np
from pathlib import Path

# Locate the ground-truth file for subject3
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "datasets" / "raw" / "subject3" / "ground_truth.txt"

# Load the three rows: BVP, HR and timestamps
data = np.loadtxt(FILE_PATH)

# Extract individual signals
bvp = data[0]
hr = data[1]
timestamps = data[2]

# Basic sample counts
print("\nSUBJECT 3 - GROUND TRUTH VALIDATION")
print("-" * 45)

print("BVP samples:", len(bvp))
print("HR samples:", len(hr))
print("Timestamp samples:", len(timestamps))

# Check whether all three signals have matching lengths
print("\nLength consistency:", len(bvp) == len(hr) == len(timestamps))

# Check for missing or infinite values
print("BVP finite:", np.isfinite(bvp).all())
print("HR finite:", np.isfinite(hr).all())
print("Timestamps finite:", np.isfinite(timestamps).all())

# Calculate timestamp differences
time_diffs = np.diff(timestamps)

# Identify duplicate and decreasing timestamps
duplicate_count = np.sum(time_diffs == 0)
decreasing_count = np.sum(time_diffs < 0)

# Calculate sampling frequency using median interval
median_interval = np.median(time_diffs[time_diffs > 0])
sampling_rate = 1 / median_interval

# Calculate recording duration
duration = timestamps[-1] - timestamps[0]

print("\nTIMESTAMP ANALYSIS")
print("-" * 45)

print("Duplicate timestamps:", duplicate_count)
print("Decreasing timestamps:", decreasing_count)
print("Median interval:", median_interval)
print("Estimated sampling rate:", round(sampling_rate, 3), "Hz")
print("Recording duration:", round(duration, 3), "seconds")

# Signal ranges
print("\nSIGNAL RANGES")
print("-" * 45)

print("BVP:", np.min(bvp), "to", np.max(bvp))
print("HR:", np.min(hr), "to", np.max(hr))

print("\nValidation inspection completed.")