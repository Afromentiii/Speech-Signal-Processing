from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


def extract_single_envelope_features(
    env: np.ndarray,
    sr: int = 32000,
    feature_mode: str = "rhythm",
    envelope_points: int = 100,
) -> dict:
    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_max = float(np.max(env))
    rms = float(np.sqrt(np.mean(env**2)))
    crest_factor = float(env_max / rms) if rms > 0 else 0.0

    diff_env = np.diff(env)
    mean_diff = float(np.mean(np.abs(diff_env)))
    std_diff = float(np.std(diff_env))

    min_peak_distance = max(1, int(sr * 0.08))
    peaks, _ = find_peaks(env, height=env_mean, distance=min_peak_distance, prominence=0.01)

    duration_sec = max(1e-4, len(env) / sr)
    if len(peaks) > 1:
        peak_intervals_sec = np.diff(peaks) / sr
        mean_peak_dist = float(np.mean(peak_intervals_sec))
        std_peak_dist = float(np.std(peak_intervals_sec))
    else:
        mean_peak_dist = 0.0
        std_peak_dist = 0.0

    peak_rate_hz = float(len(peaks) / duration_sec)

    features = {
        "mean": env_mean,
        "std": env_std,
        "max": env_max,
        "skew": float(skew(env)),
        "kurtosis": float(kurtosis(env)),
        "rms": rms,
        "crest_factor": crest_factor,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "mean_peak_dist": mean_peak_dist,
        "std_peak_dist": std_peak_dist,
        "peak_rate_hz": peak_rate_hz,
    }

    if feature_mode in ("rhythm", "advanced"):
        percentiles = np.percentile(env, [10, 25, 50, 75, 90])
        for p_val, p_num in zip(percentiles, [10, 25, 50, 75, 90]):
            features[f"p_{p_num}"] = float(p_val)

        env_zero_mean = env - env_mean
        hann_window = np.hanning(len(env))
        env_windowed = env_zero_mean * hann_window

        fft_mag = np.abs(np.fft.rfft(env_windowed))
        fft_freqs = np.fft.rfftfreq(len(env), d=1.0 / sr)

        for hz in range(1, 9):
            band_mask = (fft_freqs >= (hz - 0.5)) & (fft_freqs < (hz + 0.5))
            if np.any(band_mask):
                features[f"rhythm_{hz}hz"] = float(np.mean(fft_mag[band_mask]))
            else:
                features[f"rhythm_{hz}hz"] = 0.0

        syllabic_mask = (fft_freqs >= 2.0) & (fft_freqs <= 8.0)
        if np.any(syllabic_mask):
            syl_freqs = fft_freqs[syllabic_mask]
            syl_mags = fft_mag[syllabic_mask]
            features["rhythm_dominant_freq_hz"] = float(syl_freqs[np.argmax(syl_mags)])
        else:
            features["rhythm_dominant_freq_hz"] = 0.0

        low_mask = (fft_freqs >= 1.0) & (fft_freqs < 3.0)
        syl_energy = np.sum(fft_mag[syllabic_mask]**2) if np.any(syllabic_mask) else 0.0
        low_energy = np.sum(fft_mag[low_mask]**2) if np.any(low_mask) else 0.0
        features["rhythm_syllabic_ratio"] = float(syl_energy / (low_energy + 1e-8))

    elif feature_mode == "raw_envelope":
        indices = np.linspace(0, len(env) - 1, envelope_points, dtype=int)
        sampled_env = env[indices]
        for i, val in enumerate(sampled_env):
            features[f"env_{i:03d}"] = float(val)

    return features


def extract_envelope_features(
    envelope_list: list[np.ndarray],
    sr: int = 32000,
    feature_mode: str = "rhythm",
    envelope_points: int = 100,
) -> pd.DataFrame:
    features = [
        extract_single_envelope_features(
            env,
            sr=sr,
            feature_mode=feature_mode,
            envelope_points=envelope_points,
        )
        for env in envelope_list
    ]
    return pd.DataFrame(features)


def _process_speaker(args: tuple) -> list[dict]:
    speaker_entry, feature_mode, envelope_points = args
    results = []
    speaker = speaker_entry["speaker"]
    label = speaker_entry["label"]

    for frag_data in speaker_entry["fragments"]:
        env = frag_data["fragment"]
        sr = frag_data.get("sr", 32000)
        feat = extract_single_envelope_features(
            env,
            sr=sr,
            feature_mode=feature_mode,
            envelope_points=envelope_points,
        )
        feat["speaker"] = speaker
        feat["label"] = label
        results.append(feat)

    return results


def build_feature_dataframe(
    speakers_data: list[dict],
    feature_mode: str = "rhythm",
    envelope_points: int = 100,
    max_workers: int = 8,
) -> pd.DataFrame:
    all_samples = []
    tasks = [(s, feature_mode, envelope_points) for s in speakers_data]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for speaker_features in executor.map(_process_speaker, tasks):
            all_samples.extend(speaker_features)

    return pd.DataFrame(all_samples)
