import os
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def apply_highpass_filter(
    signal: np.ndarray,
    cutoff_hz: float = 75.0,
    sr: int = 32000,
    order: int = 4,
) -> np.ndarray:
    if cutoff_hz is None or cutoff_hz <= 0:
        return signal
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, signal)


def compute_envelope(signal: np.ndarray, window_size: int = 1000, device: str = "auto") -> np.ndarray:
    signal_abs = np.abs(signal)

    if device == "auto":
        use_cuda = HAS_TORCH and torch.cuda.is_available()
    else:
        use_cuda = HAS_TORCH and device.startswith("cuda") and torch.cuda.is_available()

    if use_cuda:
        signal_tensor = torch.from_numpy(signal_abs).float().unsqueeze(0).unsqueeze(0).cuda()
        kernel = (torch.ones(1, 1, window_size, dtype=torch.float32) / window_size).cuda()
        padding = window_size // 2
        envelope_tensor = torch.nn.functional.conv1d(signal_tensor, kernel, padding=padding)
        envelope = envelope_tensor.squeeze().cpu().numpy()
        if len(envelope) > len(signal):
            envelope = envelope[: len(signal)]
        elif len(envelope) < len(signal):
            envelope = np.pad(envelope, (0, len(signal) - len(envelope)), mode="edge")
        return envelope

    kernel = np.ones(window_size, dtype=np.float32) / np.float32(window_size)
    return np.convolve(signal_abs.astype(np.float32), kernel, mode="same").astype(np.float32)


def cut_audio(
    path: str,
    language_name: str,
    speaker_id: int,
    target_sr: int = 32000,
    window_size: int = 1000,
    silence_threshold: float = 0.002,
    highpass_cutoff: float | None = 75.0,
    device: str = "auto",
) -> dict | None:
    try:
        audio_file, sr = librosa.load(path, sr=target_sr)
    except Exception:
        return None

    if highpass_cutoff is not None and highpass_cutoff > 0:
        audio_file = apply_highpass_filter(audio_file, cutoff_hz=highpass_cutoff, sr=sr)

    step = sr
    fragments = []

    for i in range(0, len(audio_file) - step, step):
        fragment = audio_file[i : i + step]
        envelope = compute_envelope(fragment, window_size=window_size, device=device)

        if np.mean(envelope) < silence_threshold:
            continue

        fragments.append({
            "raw_signal": fragment,
            "envelope": envelope,
            "fragment": envelope,
            "sr": sr,
        })

    if len(fragments) > 0:
        return {
            "speaker": f"Speaker_{speaker_id}",
            "label": language_name,
            "fragments": fragments,
        }
    return None


def resolve_audio_folder(folder_path: str) -> str:
    clips_subfolder = os.path.join(folder_path, "clips")
    if os.path.exists(clips_subfolder) and os.path.isdir(clips_subfolder):
        return clips_subfolder
    return folder_path


def cut_and_audit_clip(
    path: str,
    language_name: str,
    target_sr: int = 32000,
    window_size: int = 1000,
    silence_threshold: float = 0.002,
    highpass_cutoff: float | None = 75.0,
    device: str = "auto",
) -> dict | None:
    try:
        audio_file, sr = librosa.load(path, sr=target_sr)
    except Exception:
        return None

    if highpass_cutoff is not None and highpass_cutoff > 0:
        audio_file = apply_highpass_filter(audio_file, cutoff_hz=highpass_cutoff, sr=sr)

    step = sr
    fragments = []
    silent_count = 0
    active_count = 0

    for i in range(0, len(audio_file) - step, step):
        fragment = np.ascontiguousarray(audio_file[i : i + step], dtype=np.float32)
        envelope = compute_envelope(fragment, window_size=window_size, device=device)
        mean_env = float(np.mean(envelope))

        if mean_env < silence_threshold:
            silent_count += 1
        else:
            active_count += 1
            fragments.append({
                "raw_signal": fragment,
                "envelope": envelope,
                "fragment": envelope,
                "sr": sr,
            })

    is_silent = len(fragments) == 0
    valid_speaker = None
    if not is_silent:
        valid_speaker = {
            "label": language_name,
            "fragments": fragments,
        }

    return {
        "valid_speaker": valid_speaker,
        "total_frags": silent_count + active_count,
        "silent_frags": silent_count,
        "active_frags": active_count,
        "duration": float(len(audio_file) / sr),
        "is_silent_clip": is_silent,
    }


def audit_and_load_dataset(
    folder_configs: dict[str, str],
    max_clips_per_language: int | None = None,
    target_sr: int = 32000,
    window_size: int = 1000,
    silence_threshold: float = 0.002,
    highpass_cutoff: float | None = 75.0,
    device: str = "auto",
    shuffle: bool = True,
    random_state: int = 42,
    max_workers: int = 8,
) -> tuple[list[dict], pd.DataFrame]:
    valid_speakers = []
    audit_rows = []

    for language_name, base_folder in folder_configs.items():
        folder = resolve_audio_folder(base_folder)
        if not os.path.exists(folder):
            continue

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg"))
        ]

        if shuffle:
            import random
            rng = random.Random(random_state)
            rng.shuffle(files)

        if max_clips_per_language is not None:
            files = files[:max_clips_per_language]

        total_clips = len(files)
        total_audio_sec = 0.0
        total_frags = 0
        silent_frags = 0
        active_frags = 0
        removed_silent_clips = 0

        import gc
        batch_size = 200
        for b_start in range(0, len(files), batch_size):
            batch_files = files[b_start : b_start + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [
                    executor.submit(
                        cut_and_audit_clip,
                        file_path,
                        language_name,
                        target_sr,
                        window_size,
                        silence_threshold,
                        highpass_cutoff,
                        device,
                    )
                    for file_path in batch_files
                ]
                for task in tasks:
                    res = task.result()
                    if res is None:
                        removed_silent_clips += 1
                        continue

                    total_audio_sec += res["duration"]
                    total_frags += res["total_frags"]
                    silent_frags += res["silent_frags"]
                    active_frags += res["active_frags"]

                    if res["is_silent_clip"] or res["valid_speaker"] is None:
                        removed_silent_clips += 1
                    else:
                        valid_speakers.append(res["valid_speaker"])
            gc.collect()

        silence_pct = (silent_frags / total_frags * 100.0) if total_frags > 0 else 0.0
        retained_speakers = total_clips - removed_silent_clips

        audit_rows.append({
            "Język": language_name,
            "Wczytane nagrania": total_clips,
            "Łączny czas audio (s)": round(total_audio_sec, 1),
            "Śr. długość nagrania (s)": round(total_audio_sec / total_clips, 2) if total_clips > 0 else 0.0,
            "Wszystkie fragmenty 1s": total_frags,
            "Aktywne fragmenty (mowa)": active_frags,
            "Odrzucona cisza (< 0.002)": silent_frags,
            "Odsetek ciszy (%)": f"{silence_pct:.2f}%",
            "Usunięte ciche nagrania (błędne)": removed_silent_clips,
            "Pozostali poprawni mówcy": retained_speakers,
        })

    for idx, sp in enumerate(valid_speakers):
        sp["speaker"] = f"Speaker_{idx}"

    audit_df = pd.DataFrame(audit_rows)
    return valid_speakers, audit_df


def load_dataset_from_folders(
    folder_configs: dict[str, str],
    max_clips_per_language: int | None = None,
    target_sr: int = 32000,
    window_size: int = 1000,
    silence_threshold: float = 0.002,
    highpass_cutoff: float | None = 75.0,
    device: str = "auto",
    max_workers: int = 8,
) -> list[dict]:
    speakers, _ = audit_and_load_dataset(
        folder_configs=folder_configs,
        max_clips_per_language=max_clips_per_language,
        target_sr=target_sr,
        window_size=window_size,
        silence_threshold=silence_threshold,
        highpass_cutoff=highpass_cutoff,
        device=device,
        max_workers=max_workers,
    )
    return speakers


def get_speaker_by_label(
    speakers_data: list[dict],
    label: str,
    index: int | None = None,
    random_state: int | None = None,
) -> dict:
    candidates = [s for s in speakers_data if s["label"] == label]
    if not candidates:
        raise ValueError(f"Brak mówców dla etykiety: {label}")
    if index is not None:
        return candidates[index % len(candidates)]
    rng = np.random.RandomState(random_state)
    return candidates[rng.randint(0, len(candidates))]

