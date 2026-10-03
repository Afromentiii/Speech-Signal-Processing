from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def compute_spectral_pooling_features(
    raw_signal: np.ndarray,
    sr: int = 32000,
    feature_type: str = "mfcc",
    n_mfcc: int = 13,
    n_mels: int = 40,
    include_higher_order_stats: bool = False,
) -> dict:
    features = {}

    if feature_type in ("mfcc", "both"):
        mfcc = librosa.feature.mfcc(y=raw_signal, sr=sr, n_mfcc=n_mfcc)
        mfcc_d1 = librosa.feature.delta(mfcc, order=1)
        mfcc_d2 = librosa.feature.delta(mfcc, order=2)

        for prefix, matrix in (("mfcc", mfcc), ("mfcc_d1", mfcc_d1), ("mfcc_d2", mfcc_d2)):
            for i in range(matrix.shape[0]):
                row = matrix[i]
                features[f"{prefix}_{i:02d}_mean"] = float(np.mean(row))
                features[f"{prefix}_{i:02d}_std"] = float(np.std(row))
                if include_higher_order_stats:
                    features[f"{prefix}_{i:02d}_skew"] = float(skew(row)) if len(row) > 2 else 0.0
                    features[f"{prefix}_{i:02d}_kurt"] = float(kurtosis(row)) if len(row) > 3 else 0.0

    if feature_type in ("mel", "both"):
        mel = librosa.feature.melspectrogram(y=raw_signal, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_d1 = librosa.feature.delta(mel_db, order=1)
        mel_d2 = librosa.feature.delta(mel_db, order=2)

        for prefix, matrix in (("mel", mel_db), ("mel_d1", mel_d1), ("mel_d2", mel_d2)):
            for i in range(matrix.shape[0]):
                row = matrix[i]
                features[f"{prefix}_{i:02d}_mean"] = float(np.mean(row))
                features[f"{prefix}_{i:02d}_std"] = float(np.std(row))
                if include_higher_order_stats:
                    features[f"{prefix}_{i:02d}_skew"] = float(skew(row)) if len(row) > 2 else 0.0
                    features[f"{prefix}_{i:02d}_kurt"] = float(kurtosis(row)) if len(row) > 3 else 0.0

    return features


def compute_prosody_features(
    raw_signal: np.ndarray,
    sr: int = 32000,
    fmin: float = 60.0,
    fmax: float = 400.0,
) -> dict:
    features = {}

    try:
        sig_yin = raw_signal[::2] if sr >= 32000 else raw_signal
        sr_yin = sr // 2 if sr >= 32000 else sr
        f0 = librosa.yin(sig_yin, fmin=fmin, fmax=fmax, sr=sr_yin, frame_length=1024, hop_length=256)
        voiced = f0[(f0 >= fmin + 5) & (f0 <= fmax - 5) & np.isfinite(f0)]
        if len(voiced) > 0:
            features["f0_mean"] = float(np.mean(voiced))
            features["f0_std"] = float(np.std(voiced))
            features["f0_min"] = float(np.min(voiced))
            features["f0_max"] = float(np.max(voiced))
            features["f0_range"] = float(features["f0_max"] - features["f0_min"])
            features["f0_skew"] = float(skew(voiced)) if len(voiced) > 2 else 0.0
            features["f0_voiced_fraction"] = float(len(voiced) / len(f0))
        else:
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0
            features["f0_min"] = 0.0
            features["f0_max"] = 0.0
            features["f0_range"] = 0.0
            features["f0_skew"] = 0.0
            features["f0_voiced_fraction"] = 0.0
    except Exception:
        features["f0_mean"] = 0.0
        features["f0_std"] = 0.0
        features["f0_min"] = 0.0
        features["f0_max"] = 0.0
        features["f0_range"] = 0.0
        features["f0_skew"] = 0.0
        features["f0_voiced_fraction"] = 0.0

    try:
        rms = librosa.feature.rms(y=raw_signal, frame_length=2048, hop_length=512)[0]
        max_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
        thresh = max(1e-4, 0.1 * max_rms)
        phonation_frames = int(np.sum(rms > thresh))
        total_frames = max(1, len(rms))
        phonation_ratio = float(phonation_frames / total_frames)
        pause_ratio = float(1.0 - phonation_ratio)

        features["phonation_ratio"] = phonation_ratio
        features["pause_ratio"] = pause_ratio
        features["pause_to_phonation_ratio"] = float(pause_ratio / (phonation_ratio + 1e-6))
    except Exception:
        features["phonation_ratio"] = 0.0
        features["pause_ratio"] = 0.0
        features["pause_to_phonation_ratio"] = 0.0

    try:
        onset_env = librosa.onset.onset_strength(y=raw_signal, sr=sr)
        tempo_val = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration_sec = max(0.1, len(raw_signal) / sr)

        features["tempo_bpm"] = tempo_val
        features["onset_rate"] = float(len(onsets) / duration_sec)
    except Exception:
        features["tempo_bpm"] = 0.0
        features["onset_rate"] = 0.0

    return features


def extract_spectral_and_prosody_features(
    raw_signal: np.ndarray,
    sr: int = 32000,
    feature_type: str = "mfcc",
    n_mfcc: int = 13,
    n_mels: int = 40,
    include_prosody: bool = True,
    include_higher_order_stats: bool = False,
) -> dict:
    feats = compute_spectral_pooling_features(
        raw_signal=raw_signal,
        sr=sr,
        feature_type=feature_type,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        include_higher_order_stats=include_higher_order_stats,
    )

    if include_prosody:
        prosody = compute_prosody_features(raw_signal=raw_signal, sr=sr)
        feats.update(prosody)

    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats


def _process_speaker_spectral(args: tuple) -> list[dict]:
    speaker_entry, feature_type, n_mfcc, n_mels, include_prosody, include_higher_order_stats = args
    results = []
    speaker = speaker_entry["speaker"]
    label = speaker_entry["label"]

    for frag_data in speaker_entry["fragments"]:
        raw_sig = frag_data.get("raw_signal")
        if raw_sig is None:
            raw_sig = frag_data.get("fragment")
        sr = frag_data.get("sr", 32000)

        feat = extract_spectral_and_prosody_features(
            raw_signal=raw_sig,
            sr=sr,
            feature_type=feature_type,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            include_prosody=include_prosody,
            include_higher_order_stats=include_higher_order_stats,
        )
        feat["speaker"] = speaker
        feat["label"] = label
        results.append(feat)

    return results


def build_spectral_feature_dataframe(
    speakers_data: list[dict],
    feature_type: str = "mfcc",
    n_mfcc: int = 13,
    n_mels: int = 40,
    include_prosody: bool = True,
    include_higher_order_stats: bool = False,
    max_workers: int = 8,
) -> pd.DataFrame:
    tasks_args = [
        (sp, feature_type, n_mfcc, n_mels, include_prosody, include_higher_order_stats)
        for sp in speakers_data
    ]

    all_rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res_list in executor.map(_process_speaker_spectral, tasks_args):
            all_rows.extend(res_list)

    df = pd.DataFrame(all_rows)
    cols = ["speaker", "label"] + [c for c in df.columns if c not in ("speaker", "label")]
    return df[cols]
