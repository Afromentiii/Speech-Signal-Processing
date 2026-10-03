import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def create_chart(
    audio_file: np.ndarray,
    samples_per_second: int,
    figsize: tuple[int, int] = (10, 2),
    title: str = "Wykres sygnału",
):
    time = np.linspace(0, len(audio_file) / samples_per_second, len(audio_file))
    plt.figure(figsize=figsize)
    plt.plot(time, audio_file)
    plt.title(title)
    plt.xlabel("Czas (sekundy)")
    plt.ylabel("Amplituda")
    plt.tight_layout()
    plt.show()


def plot_speaker_fragments(fragments_list: list[dict], max_fragments: int = 4):
    count = min(len(fragments_list), max_fragments)
    for i in range(count):
        frag_data = fragments_list[i]
        create_chart(
            frag_data["fragment"],
            frag_data["sr"],
            figsize=(10, 2),
            title=f"Fragment {i}",
        )


def plot_speaker_comparison(
    speaker_left: dict,
    speaker_right: dict,
    max_fragments: int = 4,
    show_individual_fragments: bool = True,
):
    frags_l = speaker_left["fragments"][:max_fragments]
    frags_r = speaker_right["fragments"][:max_fragments]
    sr_l = frags_l[0]["sr"] if frags_l else 32000
    sr_r = frags_r[0]["sr"] if frags_r else 32000

    sig_l_cat = np.concatenate([f["fragment"] for f in frags_l]) if frags_l else np.array([])
    sig_r_cat = np.concatenate([f["fragment"] for f in frags_r]) if frags_r else np.array([])

    time_l = np.linspace(0, len(sig_l_cat) / sr_l, len(sig_l_cat)) if len(sig_l_cat) > 0 else np.array([])
    time_r = np.linspace(0, len(sig_r_cat) / sr_r, len(sig_r_cat)) if len(sig_r_cat) > 0 else np.array([])

    n_rows = 1
    if show_individual_fragments:
        n_rows += max(len(frags_l), len(frags_r))

    fig = plt.figure(figsize=(16, 2.7 * n_rows))
    gs = fig.add_gridspec(n_rows, 2)

    ax_top_l = fig.add_subplot(gs[0, 0])
    ax_top_r = fig.add_subplot(gs[0, 1])

    if len(sig_l_cat) > 0:
        ax_top_l.plot(time_l, sig_l_cat, color="tab:blue", lw=1.2)
        ax_top_l.set_title(
            f"{speaker_left['label']} ({speaker_left['speaker']}) - {len(frags_l)} fragmenty scalone poziomo",
            fontweight="bold",
        )
        ax_top_l.set_xlabel("Czas łączny (sekundy)")
        ax_top_l.set_ylabel("Amplituda")
        max_val_l = float(np.max(sig_l_cat)) if len(sig_l_cat) > 0 else 0.1
        for k in range(1, len(frags_l)):
            ax_top_l.axvline(x=k, color="red", linestyle="--", alpha=0.6)
            ax_top_l.text(k - 0.5, max_val_l * 0.9, f"F{k-1}", color="darkblue", ha="center", fontsize=9)
        if len(frags_l) > 0:
            ax_top_l.text(len(frags_l) - 0.5, max_val_l * 0.9, f"F{len(frags_l)-1}", color="darkblue", ha="center", fontsize=9)

    if len(sig_r_cat) > 0:
        ax_top_r.plot(time_r, sig_r_cat, color="tab:orange", lw=1.2)
        ax_top_r.set_title(
            f"{speaker_right['label']} ({speaker_right['speaker']}) - {len(frags_r)} fragmenty scalone poziomo",
            fontweight="bold",
        )
        ax_top_r.set_xlabel("Czas łączny (sekundy)")
        ax_top_r.set_ylabel("Amplituda")
        max_val_r = float(np.max(sig_r_cat)) if len(sig_r_cat) > 0 else 0.1
        for k in range(1, len(frags_r)):
            ax_top_r.axvline(x=k, color="red", linestyle="--", alpha=0.6)
            ax_top_r.text(k - 0.5, max_val_r * 0.9, f"F{k-1}", color="darkred", ha="center", fontsize=9)
        if len(frags_r) > 0:
            ax_top_r.text(len(frags_r) - 0.5, max_val_r * 0.9, f"F{len(frags_r)-1}", color="darkred", ha="center", fontsize=9)

    if show_individual_fragments:
        indiv_rows = max(len(frags_l), len(frags_r))
        for r in range(indiv_rows):
            ax_l = fig.add_subplot(gs[r + 1, 0])
            ax_r = fig.add_subplot(gs[r + 1, 1])

            if r < len(frags_l):
                sig = frags_l[r]["fragment"]
                t = np.linspace(0, len(sig) / frags_l[r]["sr"], len(sig))
                ax_l.plot(t, sig, color="tab:blue")
                ax_l.set_title(f"{speaker_left['label']} ({speaker_left['speaker']}) - Fragment {r}")
                ax_l.set_xlabel("Czas (sekundy)")
                ax_l.set_ylabel("Amplituda")
            else:
                ax_l.axis("off")

            if r < len(frags_r):
                sig = frags_r[r]["fragment"]
                t = np.linspace(0, len(sig) / frags_r[r]["sr"], len(sig))
                ax_r.plot(t, sig, color="tab:orange")
                ax_r.set_title(f"{speaker_right['label']} ({speaker_right['speaker']}) - Fragment {r}")
                ax_r.set_xlabel("Czas (sekundy)")
                ax_r.set_ylabel("Amplituda")
            else:
                ax_r.axis("off")

    plt.tight_layout()
    plt.show()


def plot_group_balance(stats_df: pd.DataFrame):
    subsets = [s for s in stats_df["Zbiór"].unique() if s != "Całość"]
    if not subsets:
        subsets = list(stats_df["Zbiór"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    samples_col = "Łączny czas mowy (s)" if "Łączny czas mowy (s)" in stats_df.columns else (
        "Liczba próbek (1s)" if "Liczba próbek (1s)" in stats_df.columns else "Liczba próbek"
    )

    for subset in subsets:
        sub_data = stats_df[stats_df["Zbiór"] == subset]
        axes[0].bar(
            [f"{r['Język']} ({r['Zbiór']})" for _, r in sub_data.iterrows()],
            sub_data["Liczba mówców"],
        )
        axes[1].bar(
            [f"{r['Język']} ({r['Zbiór']})" for _, r in sub_data.iterrows()],
            sub_data[samples_col],
        )

    axes[0].set_title("Liczba mówców w grupach")
    axes[0].set_ylabel("Mówcy")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].set_title("Łączny czas mowy (s) w grupach")
    axes[1].set_ylabel("Czas mowy (s)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


def plot_feature_importances(
    model,
    feature_names: list[str] | pd.Index,
    top_n: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Ważność cech w modelu",
):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    feature_names = np.array(feature_names)
    sorted_idx = np.argsort(importances)[::-1]

    if top_n is not None:
        sorted_idx = sorted_idx[:top_n]

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.bar(range(len(sorted_idx)), importances[sorted_idx])
    plt.xticks(range(len(sorted_idx)), feature_names[sorted_idx], rotation=45, ha="right")
    plt.ylabel("Ważność")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str] | np.ndarray,
    title: str = "Macierz pomyłek",
):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_spectrogram_and_mfcc_comparison(
    speaker_left: dict,
    speaker_right: dict,
    sr: int = 32000,
    n_mels: int = 40,
    n_mfcc: int = 20,
):
    import librosa

    sig_l = speaker_left["fragments"][0].get("raw_signal", speaker_left["fragments"][0].get("fragment"))
    sig_r = speaker_right["fragments"][0].get("raw_signal", speaker_right["fragments"][0].get("fragment"))
    sr_l = speaker_left["fragments"][0].get("sr", sr)
    sr_r = speaker_right["fragments"][0].get("sr", sr)

    mel_l = librosa.feature.melspectrogram(y=sig_l, sr=sr_l, n_mels=n_mels)
    mel_l_db = librosa.power_to_db(mel_l, ref=np.max)
    mfcc_l = librosa.feature.mfcc(y=sig_l, sr=sr_l, n_mfcc=n_mfcc)
    d1_l = librosa.feature.delta(mfcc_l, order=1)

    mel_r = librosa.feature.melspectrogram(y=sig_r, sr=sr_r, n_mels=n_mels)
    mel_r_db = librosa.power_to_db(mel_r, ref=np.max)
    mfcc_r = librosa.feature.mfcc(y=sig_r, sr=sr_r, n_mfcc=n_mfcc)
    d1_r = librosa.feature.delta(mfcc_r, order=1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")

    im0 = axes[0, 0].imshow(mel_l_db, aspect="auto", origin="lower", cmap="magma")
    axes[0, 0].set_title(f"{speaker_left['label']} ({speaker_left['speaker']}) - Mel-spektrogram (dB)")
    axes[0, 0].set_ylabel("Pasma Mel")
    fig.colorbar(im0, ax=axes[0, 0], format="%+2.0f dB")

    im1 = axes[0, 1].imshow(mel_r_db, aspect="auto", origin="lower", cmap="magma")
    axes[0, 1].set_title(f"{speaker_right['label']} ({speaker_right['speaker']}) - Mel-spektrogram (dB)")
    axes[0, 1].set_ylabel("Pasma Mel")
    fig.colorbar(im1, ax=axes[0, 1], format="%+2.0f dB")

    im2 = axes[1, 0].imshow(mfcc_l, aspect="auto", origin="lower", cmap="coolwarm")
    axes[1, 0].set_title(f"{speaker_left['label']} - MFCC ({n_mfcc} pasm)")
    axes[1, 0].set_ylabel("Współczynniki")
    fig.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(mfcc_r, aspect="auto", origin="lower", cmap="coolwarm")
    axes[1, 1].set_title(f"{speaker_right['label']} - MFCC ({n_mfcc} pasm)")
    axes[1, 1].set_ylabel("Współczynniki")
    fig.colorbar(im3, ax=axes[1, 1])

    im4 = axes[2, 0].imshow(d1_l, aspect="auto", origin="lower", cmap="viridis")
    axes[2, 0].set_title(f"{speaker_left['label']} - Pochodna (\u0394 MFCC)")
    axes[2, 0].set_xlabel("Ramki czasowe")
    axes[2, 0].set_ylabel("Pochodne \u0394")
    fig.colorbar(im4, ax=axes[2, 0])

    im5 = axes[2, 1].imshow(d1_r, aspect="auto", origin="lower", cmap="viridis")
    axes[2, 1].set_title(f"{speaker_right['label']} - Pochodna (\u0394 MFCC)")
    axes[2, 1].set_xlabel("Ramki czasowe")
    axes[2, 1].set_ylabel("Pochodne \u0394")
    fig.colorbar(im5, ax=axes[2, 1])

    plt.tight_layout()
    plt.show()


def plot_svm_grid_search(cv_results_df: pd.DataFrame):
    c_col = "param_svm__C" if "param_svm__C" in cv_results_df.columns else (
        "param_C" if "param_C" in cv_results_df.columns else None
    )
    g_col = "param_svm__gamma" if "param_svm__gamma" in cv_results_df.columns else (
        "param_gamma" if "param_gamma" in cv_results_df.columns else None
    )

    if not c_col or not g_col:
        return

    df_filtered = cv_results_df
    if "param_svm__kernel" in cv_results_df.columns:
        df_filtered = cv_results_df[cv_results_df["param_svm__kernel"] == "rbf"]
    elif "param_kernel" in cv_results_df.columns:
        df_filtered = cv_results_df[cv_results_df["param_kernel"] == "rbf"]

    if df_filtered.empty:
        df_filtered = cv_results_df

    pivot = df_filtered.pivot_table(index=c_col, columns=g_col, values="mean_test_score", aggfunc="mean")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, label="Dokładność CV")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [str(r) for r in pivot.index])
    plt.xlabel("Parametr \u03b3 (gamma)")
    plt.ylabel("Parametr C")
    plt.title("Grid Search SVM: Dokładność CV dla C vs \u03b3")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                text_color = "white" if val > 0.8 else "black"
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    pos_label: str | int = 1,
    title: str = "Krzywa ROC",
):
    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_predictions(y_true, y_prob, pos_label=pos_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_cv_stability(fold_df: pd.DataFrame, figsize: tuple[int, int] = (12, 4)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    folds = fold_df["Fold"]
    train_acc = fold_df["Train Accuracy"]
    val_acc = fold_df["Val Accuracy"]

    axes[0].plot(folds, train_acc, "o-", color="tab:blue", label="Trening (Train Acc)")
    axes[0].plot(folds, val_acc, "s--", color="tab:orange", label="Walidacja (Val Acc)")
    axes[0].fill_between(folds, train_acc, val_acc, color="gray", alpha=0.15, label="Luka generalizacji")
    axes[0].axhline(y=float(val_acc.mean()), color="darkred", linestyle=":", label=f"Średnia walidacji: {val_acc.mean():.2%}")
    axes[0].set_xlabel("Numer foldu")
    axes[0].set_ylabel("Dokładność (Accuracy)")
    axes[0].set_title("Stabilność dokładności pomiędzy foldami")
    axes[0].set_xticks(folds)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    metrics = ["Val Accuracy", "Val Precision", "Val Recall", "Val F1"]
    if "Val ROC-AUC" in fold_df.columns and fold_df["Val ROC-AUC"].notna().any():
        metrics.append("Val ROC-AUC")

    sub_data = fold_df[["Fold"] + metrics].set_index("Fold")
    sub_data.plot(kind="bar", ax=axes[1], colormap="viridis", width=0.8)
    axes[1].set_xlabel("Numer foldu")
    axes[1].set_ylabel("Wynik metryki")
    axes[1].set_title("Rozkład metryk na poszczególnych foldach")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_feature_stability(feature_stability_df: pd.DataFrame, top_n: int = 15, figsize: tuple[int, int] = (10, 5)):
    if feature_stability_df is None or feature_stability_df.empty:
        return

    top_df = feature_stability_df.head(top_n).sort_values(by="Średnia ważność", ascending=True)

    plt.figure(figsize=figsize)
    plt.barh(top_df["Cecha"], top_df["Średnia ważność"], xerr=top_df["Odch. std"], capsize=4, color="teal", alpha=0.85)
    plt.xlabel("Średnia ważność cechy")
    plt.title(f"Stabilność ważności cech w walidacji krzyżowej (Top {len(top_df)})")
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_multiseed_stability(seed_df: pd.DataFrame, figsize: tuple[int, int] = (12, 4)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    x_indices = np.arange(len(seed_df))
    seed_labels = [f"Seed {s}" for s in seed_df["Seed"]]
    tr_acc = seed_df["Train Accuracy"]
    te_acc = seed_df["Test Accuracy"]

    axes[0].plot(x_indices, tr_acc, "o-", color="tab:blue", label="Trening (Train Acc)")
    axes[0].plot(x_indices, te_acc, "s--", color="tab:orange", label="Test (Test Acc)")
    axes[0].fill_between(x_indices, tr_acc, te_acc, color="gray", alpha=0.15, label="Luka generalizacji")
    axes[0].axhline(y=float(te_acc.mean()), color="darkred", linestyle=":", label=f"Średnia testu: {te_acc.mean():.2%}")
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(seed_labels, rotation=30)
    axes[0].set_ylabel("Dokładność (Accuracy)")
    axes[0].set_title("Dokładność modelu dla różnych ziaren losowości")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    metrics = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
    if "Test ROC-AUC" in seed_df.columns and seed_df["Test ROC-AUC"].notna().any():
        metrics.append("Test ROC-AUC")

    plot_data = seed_df.set_index("Seed")[metrics]
    plot_data.plot(kind="bar", ax=axes[1], colormap="viridis", width=0.8)
    axes[1].set_xlabel("Ziarno losowości (Seed)")
    axes[1].set_ylabel("Wynik metryki")
    axes[1].set_title("Rozkład metryk testowych w zależności od ziarna")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
