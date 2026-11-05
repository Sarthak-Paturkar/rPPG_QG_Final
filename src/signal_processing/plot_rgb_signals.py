import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate project files
ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "datasets/processed/subject3/rgb_signals.csv"

OUTPUT_DIR = ROOT / "outputs/plots/subject3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load extracted RGB measurements
df = pd.read_csv(CSV_PATH)

# Define the three facial regions
regions = {
    "Forehead": "forehead",
    "Right Cheek": "right_cheek",
    "Left Cheek": "left_cheek"
}

channels = ["r", "g", "b"]

# Plot RGB signals separately for each region
for region_name, prefix in regions.items():

    plt.figure(figsize=(14, 5))

    for channel in channels:

        column = f"{prefix}_{channel}"

        plt.plot(
            df["video_time_seconds"],
            df[column],
            label=channel.upper(),
            linewidth=0.8
        )

    plt.title(f"{region_name} - Raw RGB Signals")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mean Pixel Intensity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save each region's graph
    output_file = OUTPUT_DIR / f"{prefix}_rgb.png"

    plt.savefig(output_file, dpi=200)
    plt.close()

    print("Saved:", output_file)

print("\nRGB visualisation completed.")