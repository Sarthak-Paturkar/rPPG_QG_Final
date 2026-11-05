import unittest

import numpy as np
from scipy import signal

from src.preprocessing.preprocess_signals import (
    DEFAULT_SETTINGS,
    infer_sampling_rate,
    preprocess_channel,
)


class PreprocessSignalsTests(unittest.TestCase):
    def test_infers_rate_from_timestamps_and_rejects_jitter(self) -> None:
        timestamps = np.arange(600, dtype=float) / 30.0
        rate, interval = infer_sampling_rate(
            timestamps, DEFAULT_SETTINGS.timestamp_relative_tolerance
        )
        self.assertAlmostEqual(rate, 30.0)
        self.assertAlmostEqual(interval, 1 / 30.0)

        irregular = timestamps.copy()
        irregular[300] += 0.002
        with self.assertRaisesRegex(ValueError, "not uniformly sampled"):
            infer_sampling_rate(
                irregular, DEFAULT_SETTINGS.timestamp_relative_tolerance
            )

    def test_bandpass_retains_pulse_and_attenuates_out_of_band_tone(self) -> None:
        sampling_rate = 30.0
        times = np.arange(900, dtype=float) / sampling_rate
        pulse = np.sin(2 * np.pi * 1.2 * times)
        noise = 0.8 * np.sin(2 * np.pi * 6.0 * times)
        values = 100.0 + 0.02 * times + pulse + noise
        stages = preprocess_channel(values, np.ones(values.size, dtype=bool), sampling_rate)

        frequencies, power = signal.welch(
            stages.normalized, fs=sampling_rate, nperseg=600
        )
        pulse_power = float(power[np.argmin(np.abs(frequencies - 1.2))])
        out_of_band_power = float(power[np.argmin(np.abs(frequencies - 6.0))])
        self.assertGreater(pulse_power, 0)
        self.assertLess(out_of_band_power, pulse_power * 0.01)
        self.assertAlmostEqual(float(np.mean(stages.normalized)), 0.0, places=10)
        self.assertAlmostEqual(float(np.std(stages.normalized)), 1.0, places=10)

    def test_interpolates_short_internal_gaps_but_preserves_long_gaps(self) -> None:
        sampling_rate = 30.0
        times = np.arange(600, dtype=float) / sampling_rate
        values = np.sin(2 * np.pi * 1.2 * times)
        valid = np.ones(values.size, dtype=bool)
        valid[100:102] = False
        valid[300:305] = False

        stages = preprocess_channel(values, valid, sampling_rate)

        self.assertEqual(int(stages.interpolated_mask.sum()), 2)
        self.assertTrue(np.isfinite(stages.filled[100:102]).all())
        self.assertTrue(np.isnan(stages.filled[300:305]).all())
        self.assertTrue(np.isnan(stages.normalized[300:305]).all())
        self.assertEqual(stages.filtered_segment_count, 2)


if __name__ == "__main__":
    unittest.main()
