import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def check_gpu_availability() -> dict:
    gpu_info = {
        "cuda_available": False,
        "device_name": None,
        "backend": None,
    }

    if HAS_TORCH and torch.cuda.is_available():
        gpu_info["cuda_available"] = True
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_info["backend"] = "PyTorch CUDA"
        return gpu_info

    if HAS_XGBOOST:
        try:
            clf = xgb.XGBClassifier(n_estimators=1, tree_method="hist", device="cuda")
            clf.fit(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0, 1]))
            gpu_info["cuda_available"] = True
            gpu_info["device_name"] = "NVIDIA CUDA Device"
            gpu_info["backend"] = "XGBoost CUDA"
            return gpu_info
        except Exception:
            pass

    return gpu_info


def create_svm_classifier(
    kernel: str = "rbf",
    C: float = 0.5,
    gamma: str | float = "scale",
    use_pca: bool = True,
    n_components: int | None = 20,
    probability: bool = True,
    random_state: int = 42,
    **kwargs,
) -> Pipeline | SVC:
    svc_params = {
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
        "probability": probability,
        "random_state": random_state,
    }
    svc_params.update(kwargs)
    svc = SVC(**svc_params)

    if use_pca:
        steps = [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
            ("svm", svc),
        ]
        return Pipeline(steps)
    return svc


def tune_svm_grid_search(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray | None = None,
    param_grid: dict | None = None,
    use_pca: bool = True,
    n_components: int | None = 20,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[Pipeline, dict, pd.DataFrame]:
    steps = [("scaler", StandardScaler())]
    if use_pca:
        pca_comp = n_components if n_components is not None else min(20, X_train.shape[1])
        steps.append(("pca", PCA(n_components=pca_comp, random_state=random_state)))
    steps.append(("svm", SVC(probability=False, random_state=random_state)))
    pipeline = Pipeline(steps)

    if param_grid is None:
        param_grid = {
            "svm__C": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            "svm__gamma": [0.001, 0.005, 0.01, 0.02, 0.05, "scale"],
            "svm__kernel": ["rbf", "linear"],
        }

    if groups is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring="accuracy",
        n_jobs=2 if n_jobs == -1 else n_jobs,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train, groups=groups)

    best_estimator = clone(grid_search.best_estimator_)
    best_estimator.named_steps["svm"].probability = True
    best_estimator.fit(X_train, y_train)

    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    return best_estimator, grid_search.best_params_, cv_results_df


def tune_svm_optuna(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray | None = None,
    use_pca: bool = True,
    n_components: int | None = 20,
    n_trials: int = 30,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[Pipeline, dict, pd.DataFrame]:
    import optuna
    from sklearn.model_selection import cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if groups is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv

    def objective(trial):
        c = trial.suggest_float("C", 0.02, 10.0, log=True)
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        gamma = trial.suggest_float("gamma", 0.0005, 0.1, log=True) if kernel == "rbf" else "scale"

        steps = [("scaler", StandardScaler())]
        if use_pca:
            steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
        steps.append(("svm", SVC(kernel=kernel, C=c, gamma=gamma, probability=True, random_state=random_state)))
        pipe = Pipeline(steps)

        scores = cross_val_score(pipe, X_train, y_train, groups=groups, cv=cv_splitter, scoring="accuracy", n_jobs=-1)
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_c = study.best_params["C"]
    best_kernel = study.best_params["kernel"]
    best_gamma = study.best_params.get("gamma", "scale")

    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
    steps.append(("svm", SVC(kernel=best_kernel, C=best_c, gamma=best_gamma, probability=True, random_state=random_state)))
    best_model = Pipeline(steps)
    best_model.fit(X_train, y_train)

    trials_df = study.trials_dataframe()
    return best_model, study.best_params, trials_df


def create_classifier(model_type: str = "auto", random_state: int = 42, **kwargs):
    if model_type in ("svm", "svc"):
        return create_svm_classifier(random_state=random_state, **kwargs)
    gpu_status = check_gpu_availability()

    if model_type in ("auto", "gpu", "xgboost") and HAS_XGBOOST:
        device = "cuda" if gpu_status["cuda_available"] else "cpu"
        default_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 8,
            "gamma": 0.5,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": device,
            "random_state": random_state,
            "eval_metric": "logloss",
        }
        default_params.update(kwargs)
        return xgb.XGBClassifier(**default_params)

    default_rf_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_leaf": 10,
        "random_state": random_state,
        "n_jobs": -1,
    }
    default_rf_params.update(kwargs)
    return RandomForestClassifier(**default_rf_params)


def train_and_evaluate(
    model,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    scale_features: bool = True,
) -> dict:
    is_pipe = isinstance(model, Pipeline)
    has_internal_scaler = is_pipe and any("scale" in name for name in model.named_steps.keys())

    scaler = None
    if scale_features and not has_internal_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr = X_train
        X_te = X_test

    label_encoder = None
    y_tr_encoded = y_train
    y_te_encoded = y_test

    if isinstance(y_train.iloc[0] if hasattr(y_train, "iloc") else y_train[0], str):
        label_encoder = LabelEncoder()
        y_tr_encoded = label_encoder.fit_transform(y_train)
        y_te_encoded = label_encoder.transform(y_test)

    model.fit(X_tr, y_tr_encoded)

    y_pred_tr_raw = model.predict(X_tr)
    y_pred_te_raw = model.predict(X_te)

    if label_encoder is not None:
        y_pred_tr = label_encoder.inverse_transform(y_pred_tr_raw)
        y_pred_te = label_encoder.inverse_transform(y_pred_te_raw)
        classes = label_encoder.classes_
    else:
        y_pred_tr = y_pred_tr_raw
        y_pred_te = y_pred_te_raw
        classes = np.unique(y_train)

    acc_train = float(accuracy_score(y_train, y_pred_tr))
    acc_test = float(accuracy_score(y_test, y_pred_te))

    report_train = classification_report(y_train, y_pred_tr, zero_division=0)
    report_test = classification_report(y_test, y_pred_te, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_te, labels=classes)

    final_est = model.steps[-1][1] if is_pipe else model
    importances = getattr(final_est, "feature_importances_", None)
    roc_auc = None
    y_prob_te = None

    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_te)
            y_prob_te = probas
            if probas.shape[1] == 2:
                roc_auc = float(roc_auc_score(y_te_encoded, probas[:, 1]))
        except Exception:
            pass

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "classes": classes,
        "y_pred_train": y_pred_tr,
        "y_pred_test": y_pred_te,
        "y_prob_test": y_prob_te,
        "roc_auc": roc_auc,
        "acc_train": acc_train,
        "acc_test": acc_test,
        "report_train": report_train,
        "report_test": report_test,
        "confusion_matrix": cm,
        "feature_importances": importances,
    }


def analyze_model_stability(
    fold_df: pd.DataFrame,
    fold_importances: list | None = None,
    feature_names: pd.Index | list | None = None,
) -> dict:
    val_acc = fold_df["Val Accuracy"]
    gap = fold_df["Generalization Gap"]
    val_f1 = fold_df["Val F1"]

    acc_mean = float(val_acc.mean())
    acc_std = float(val_acc.std())
    acc_cv = float(acc_std / acc_mean) if acc_mean > 0 else 0.0

    gap_mean = float(gap.mean())
    gap_std = float(gap.std())

    f1_mean = float(val_f1.mean())
    f1_std = float(val_f1.std())
    f1_cv = float(f1_std / f1_mean) if f1_mean > 0 else 0.0

    if gap_mean < 0.08 and acc_cv < 0.10:
        status = "Wysoka stabilność (brak przeuczenia)"
    elif gap_mean < 0.16 and acc_cv < 0.16:
        status = "Dobra stabilność (niska tendencja do przeuczenia)"
    else:
        status = "Niska stabilność / Istotna luka generalizacji"

    metrics_df = pd.DataFrame([
        {"Wskaźnik": "Średnia dokładność walidacji (Val Acc)", "Wartość": f"{acc_mean:.2%}"},
        {"Wskaźnik": "Odch. std walidacji (Val Acc Std)", "Wartość": f"{acc_std:.2%}"},
        {"Wskaźnik": "Współczynnik zmienności dokładności (CV = std/mean)", "Wartość": f"{acc_cv:.4f}"},
        {"Wskaźnik": "Średnia luka generalizacji (Train - Val)", "Wartość": f"{gap_mean:.2%}"},
        {"Wskaźnik": "Odch. std luki generalizacji (Gap Std)", "Wartość": f"{gap_std:.2%}"},
        {"Wskaźnik": "Średni wynik F1 (Val F1)", "Wartość": f"{f1_mean:.2%}"},
        {"Wskaźnik": "Współczynnik zmienności F1 (CV)", "Wartość": f"{f1_cv:.4f}"},
        {"Wskaźnik": "Ocena stabilności modelu", "Wartość": status},
    ])

    feat_stability_df = None
    if fold_importances and len(fold_importances) > 1 and feature_names is not None:
        imp_matrix = np.array(fold_importances)
        feat_means = np.mean(imp_matrix, axis=0)
        feat_stds = np.std(imp_matrix, axis=0)
        feat_stability_df = pd.DataFrame({
            "Cecha": feature_names,
            "Średnia ważność": feat_means,
            "Odch. std": feat_stds,
            "Współczynnik zmienności": np.where(feat_means > 0, feat_stds / (feat_means + 1e-8), 0.0),
        }).sort_values(by="Średnia ważność", ascending=False).reset_index(drop=True)

    return {
        "metrics": metrics_df,
        "features": feat_stability_df,
    }


def run_group_cross_validation(
    model,
    X: pd.DataFrame | np.ndarray | None = None,
    y: pd.Series | np.ndarray | None = None,
    groups: pd.Series | np.ndarray | None = None,
    dataset: pd.DataFrame | None = None,
    n_splits: int = 5,
    scale_features: bool = True,
    random_state: int = 42,
) -> dict:
    if dataset is not None:
        metadata_cols = {"label", "speaker", "clip_id", "filename"}
        feature_cols = [c for c in dataset.columns if c not in metadata_cols]
        X = dataset[feature_cols]
        y = dataset["label"]
        groups = dataset["speaker"]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.reset_index(drop=True)
    y_ser = pd.Series(y) if isinstance(y, np.ndarray) else y.reset_index(drop=True)
    groups_arr = np.asarray(groups)

    fold_records = []
    fold_importances = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_df, y_ser, groups_arr)):
        X_tr = X_df.iloc[train_idx]
        y_tr = y_ser.iloc[train_idx]
        X_val = X_df.iloc[val_idx]
        y_val = y_ser.iloc[val_idx]

        model_fold = clone(model)
        eval_res = train_and_evaluate(
            model=model_fold,
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_val,
            y_test=y_val,
            scale_features=scale_features,
        )

        acc_tr = eval_res["acc_train"]
        acc_val = eval_res["acc_test"]
        gap = acc_tr - acc_val
        y_pred = eval_res["y_pred_test"]

        pos_label = eval_res["classes"][1] if len(eval_res["classes"]) == 2 else None
        prec = float(precision_score(y_val, y_pred, pos_label=pos_label, average="binary" if pos_label else "weighted", zero_division=0))
        rec = float(recall_score(y_val, y_pred, pos_label=pos_label, average="binary" if pos_label else "weighted", zero_division=0))
        f1 = float(f1_score(y_val, y_pred, pos_label=pos_label, average="binary" if pos_label else "weighted", zero_division=0))
        auc = eval_res["roc_auc"]

        fold_records.append({
            "Fold": fold_idx + 1,
            "Trening Mówcy": len(np.unique(groups_arr[train_idx])),
            "Walidacja Mówcy": len(np.unique(groups_arr[val_idx])),
            "Train Accuracy": acc_tr,
            "Val Accuracy": acc_val,
            "Generalization Gap": gap,
            "Val Precision": prec,
            "Val Recall": rec,
            "Val F1": f1,
            "Val ROC-AUC": auc if auc is not None else np.nan,
        })

        if eval_res["feature_importances"] is not None:
            fold_importances.append(eval_res["feature_importances"])

    fold_df = pd.DataFrame(fold_records)

    numeric_cols = ["Train Accuracy", "Val Accuracy", "Generalization Gap", "Val Precision", "Val Recall", "Val F1"]
    if fold_df["Val ROC-AUC"].notna().any():
        numeric_cols.append("Val ROC-AUC")

    summary_records = []
    for col in numeric_cols:
        vals = fold_df[col].dropna()
        mean_val = float(vals.mean())
        std_val = float(vals.std())
        cv_metric = (std_val / mean_val) if mean_val > 0 else 0.0
        summary_records.append({
            "Metryka": col,
            "Średnia (Mean)": round(mean_val, 4),
            "Odch. std (Std)": round(std_val, 4),
            "Min": round(float(vals.min()), 4),
            "Max": round(float(vals.max()), 4),
            "Współczynnik zmienności (CV)": round(cv_metric, 4),
        })

    summary_df = pd.DataFrame(summary_records)
    stability_stats = analyze_model_stability(fold_df, fold_importances, feature_names=X_df.columns)

    return {
        "fold_results": fold_df,
        "summary": summary_df,
        "stability_analysis": stability_stats["metrics"],
        "feature_stability": stability_stats["features"],
    }


def run_multiseed_experiment(
    model,
    dataset: pd.DataFrame | None = None,
    seeds: list[int] | tuple[int, ...] = (42, 100, 2024, 7, 999),
    test_size: float = 0.25,
    scale_features: bool = True,
    X: pd.DataFrame | np.ndarray | None = None,
    y: pd.Series | np.ndarray | None = None,
    speakers: pd.Series | np.ndarray | None = None,
) -> dict:
    from .data_balancing import stratified_speaker_train_test_split

    if dataset is None:
        if X is None or y is None or speakers is None:
            raise ValueError("Należy przekazać 'dataset' lub trójkę ('X', 'y', 'speakers').")
        dataset = pd.DataFrame(X).copy()
        dataset["label"] = np.asarray(y)
        dataset["speaker"] = np.asarray(speakers)

    records = []

    for seed in seeds:
        X_tr, y_tr, X_te, y_te, sp_tr, sp_te, _, _ = stratified_speaker_train_test_split(
            dataset,
            test_size=test_size,
            random_state=seed,
        )

        model_seed = clone(model)
        res = train_and_evaluate(
            model=model_seed,
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            scale_features=scale_features,
        )

        acc_tr = res["acc_train"]
        acc_te = res["acc_test"]
        gap = acc_tr - acc_te
        y_pred = res["y_pred_test"]
        pos_label = res["classes"][1] if len(res["classes"]) == 2 else None

        prec = float(precision_score(y_te, y_pred, pos_label=pos_label, average="binary" if pos_label else "weighted", zero_division=0))
        rec = float(recall_score(y_te, y_pred, pos_label=pos_label, average="binary" if pos_label else "weighted", zero_division=0))
        f1 = float(f1_score(y_te, y_pred, pos_label=pos_label, average="binary" if pos_label else "weighted", zero_division=0))
        auc = res["roc_auc"]

        records.append({
            "Seed": seed,
            "Liczba mówców Train": len(sp_tr),
            "Liczba mówców Test": len(sp_te),
            "Train Accuracy": acc_tr,
            "Test Accuracy": acc_te,
            "Generalization Gap": gap,
            "Test Precision": prec,
            "Test Recall": rec,
            "Test F1": f1,
            "Test ROC-AUC": auc if auc is not None else np.nan,
        })

    df_seeds = pd.DataFrame(records)

    numeric_cols = ["Train Accuracy", "Test Accuracy", "Generalization Gap", "Test Precision", "Test Recall", "Test F1"]
    if df_seeds["Test ROC-AUC"].notna().any():
        numeric_cols.append("Test ROC-AUC")

    summary_records = []
    for col in numeric_cols:
        vals = df_seeds[col].dropna()
        m = float(vals.mean())
        s = float(vals.std())
        cv_val = float(s / m) if m > 0 else 0.0
        summary_records.append({
            "Metryka": col,
            "Średnia (Mean)": round(m, 4),
            "Odch. std (Std)": round(s, 4),
            "Min": round(float(vals.min()), 4),
            "Max": round(float(vals.max()), 4),
            "Współczynnik zmienności (CV)": round(cv_val, 4),
        })
    summary_df = pd.DataFrame(summary_records)

    test_acc = df_seeds["Test Accuracy"]
    test_f1 = df_seeds["Test F1"]
    gap = df_seeds["Generalization Gap"]

    m_acc, s_acc = float(test_acc.mean()), float(test_acc.std())
    m_f1, s_f1 = float(test_f1.mean()), float(test_f1.std())
    m_gap, s_gap = float(gap.mean()), float(gap.std())
    cv_acc = s_acc / m_acc if m_acc > 0 else 0.0

    if s_acc < 0.05 and m_gap < 0.12:
        status = "Wysoka odporność na ziarno losowości"
    elif s_acc < 0.09 and m_gap < 0.18:
        status = "Dobra odporność na ziarno losowości"
    else:
        status = "Umiarkowana odporność / Wrażliwość na losowy podział"

    ci_low = max(0.0, m_acc - 2 * s_acc)
    ci_high = min(1.0, m_acc + 2 * s_acc)

    stability_report = pd.DataFrame([
        {"Wskaźnik stabilności między ziarnami": "Liczba zbadanych ziaren losowości (Seeds)", "Wartość": str(len(seeds))},
        {"Wskaźnik stabilności między ziarnami": "Średnia dokładność testowa (Mean Test Acc)", "Wartość": f"{m_acc:.2%}"},
        {"Wskaźnik stabilności między ziarnami": "Odchylenie standardowe dokładności (Std)", "Wartość": f"{s_acc:.2%}"},
        {"Wskaźnik stabilności między ziarnami": "Przedział ufności ok. 95% (Mean ± 2 Std)", "Wartość": f"[{ci_low:.2%}; {ci_high:.2%}]"},
        {"Wskaźnik stabilności między ziarnami": "Współczynnik zmienności dokładności (CV)", "Wartość": f"{cv_acc:.4f}"},
        {"Wskaźnik stabilności między ziarnami": "Średni wynik F1 na teście (Mean F1)", "Wartość": f"{m_f1:.2%}"},
        {"Wskaźnik stabilności między ziarnami": "Odchylenie F1 (Std)", "Wartość": f"{s_f1:.2%}"},
        {"Wskaźnik stabilności między ziarnami": "Średnia luka generalizacji (Train - Test)", "Wartość": f"{m_gap:.2%}"},
        {"Wskaźnik stabilności między ziarnami": "Ocena odporności na losowość", "Wartość": status},
    ])

    return {
        "seed_results": df_seeds,
        "summary": summary_df,
        "stability_report": stability_report,
    }


