import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def balance_speaker_groups(
    dataset_ml: pd.DataFrame,
    min_samples_per_speaker: int | None = None,
    max_samples_per_speaker: int | None = None,
    balance_speakers_per_class: bool = True,
    balance_samples_per_class: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    df = dataset_ml.copy()

    if min_samples_per_speaker is not None and min_samples_per_speaker > 0:
        speaker_counts = df["speaker"].value_counts()
        valid_speakers = speaker_counts[speaker_counts >= min_samples_per_speaker].index
        df = df[df["speaker"].isin(valid_speakers)].reset_index(drop=True)

    if max_samples_per_speaker is not None and max_samples_per_speaker > 0:
        balanced_indices = []
        for _, group in df.groupby("speaker"):
            if len(group) <= max_samples_per_speaker:
                balanced_indices.extend(group.index.tolist())
            else:
                sampled = group.sample(n=max_samples_per_speaker, random_state=rng)
                balanced_indices.extend(sampled.index.tolist())
        df = df.loc[balanced_indices].reset_index(drop=True)

    if balance_speakers_per_class:
        speaker_labels = df.groupby("speaker")["label"].first()
        min_speakers = speaker_labels.value_counts().min()

        selected_speakers = []
        for label in speaker_labels.unique():
            speakers_in_class = speaker_labels[speaker_labels == label].index.tolist()
            chosen = rng.choice(speakers_in_class, size=min_speakers, replace=False)
            selected_speakers.extend(chosen)

        df = df[df["speaker"].isin(selected_speakers)].reset_index(drop=True)

    if balance_samples_per_class:
        class_counts = df["label"].value_counts()
        min_samples = class_counts.min()

        balanced_rows = []
        for label in class_counts.index:
            class_df = df[df["label"] == label]
            if len(class_df) > min_samples:
                class_df = class_df.sample(n=min_samples, random_state=rng)
            balanced_rows.append(class_df)

        df = pd.concat(balanced_rows, ignore_index=True)

    return df


def stratified_speaker_train_test_split(
    dataset_ml: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    speaker_labels = dataset_ml.groupby("speaker")["label"].first()
    unique_speakers = speaker_labels.index.to_numpy()
    labels_per_speaker = speaker_labels.to_numpy()

    speakers_train, speakers_test = train_test_split(
        unique_speakers,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_per_speaker,
        shuffle=True,
    )

    train_data = dataset_ml[dataset_ml["speaker"].isin(speakers_train)].reset_index(drop=True)
    test_data = dataset_ml[dataset_ml["speaker"].isin(speakers_test)].reset_index(drop=True)

    feature_cols = [c for c in dataset_ml.columns if c not in ("label", "speaker")]

    X_train = train_data[feature_cols]
    y_train = train_data["label"]

    X_test = test_data[feature_cols]
    y_test = test_data["label"]

    return X_train, y_train, X_test, y_test, speakers_train, speakers_test, train_data, test_data


def get_group_statistics(
    dataset_ml: pd.DataFrame,
    speakers_train: np.ndarray | None = None,
    speakers_test: np.ndarray | None = None,
    sample_duration_sec: float = 1.0,
) -> pd.DataFrame:
    records = []

    def _stats_for_subset(subset_df: pd.DataFrame, subset_name: str):
        speaker_labels = subset_df.groupby("speaker")["label"].first()
        samples_per_label = subset_df["label"].value_counts()
        speakers_per_label = speaker_labels.value_counts()
        avg_samples = subset_df.groupby(["label", "speaker"]).size().groupby("label").mean()

        for label in samples_per_label.index:
            n_samples = int(samples_per_label.get(label, 0))
            n_speakers = int(speakers_per_label.get(label, 0))
            total_sec = round(n_samples * sample_duration_sec, 2)
            avg_sec = round(float(avg_samples.get(label, 0.0)) * sample_duration_sec, 2)

            records.append({
                "Zbiór": subset_name,
                "Język": label,
                "Liczba mówców": n_speakers,
                "Liczba próbek (1s)": n_samples,
                "Łączny czas mowy (s)": total_sec,
                "Śr. czas / mówca (s)": avg_sec,
            })

    _stats_for_subset(dataset_ml, "Całość")

    if speakers_train is not None and speakers_test is not None:
        train_df = dataset_ml[dataset_ml["speaker"].isin(speakers_train)]
        test_df = dataset_ml[dataset_ml["speaker"].isin(speakers_test)]
        _stats_for_subset(train_df, "Trening")
        _stats_for_subset(test_df, "Test")

    return pd.DataFrame(records)
