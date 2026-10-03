import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from .audio_processing import (
    apply_highpass_filter,
    audit_and_load_dataset,
    compute_envelope,
    cut_audio,
    get_speaker_by_label,
    load_dataset_from_folders,
    resolve_audio_folder,
)
from .feature_extraction import extract_envelope_features, build_feature_dataframe
from .spectral_features import (
    compute_spectral_pooling_features,
    compute_prosody_features,
    extract_spectral_and_prosody_features,
    build_spectral_feature_dataframe,
)
from .data_balancing import balance_speaker_groups, stratified_speaker_train_test_split, get_group_statistics
from .modeling import (
    check_gpu_availability,
    create_classifier,
    create_svm_classifier,
    tune_svm_grid_search,
    tune_svm_optuna,
    train_and_evaluate,
    run_group_cross_validation,
    analyze_model_stability,
    run_multiseed_experiment,
)
from .visualization import (
    create_chart,
    plot_speaker_fragments,
    plot_speaker_comparison,
    plot_group_balance,
    plot_feature_importances,
    plot_confusion_matrix,
    plot_spectrogram_and_mfcc_comparison,
    plot_svm_grid_search,
    plot_roc_curve,
    plot_cv_stability,
    plot_feature_stability,
    plot_multiseed_stability,
)

__all__ = [
    "apply_highpass_filter",
    "audit_and_load_dataset",
    "compute_envelope",
    "cut_audio",
    "get_speaker_by_label",
    "load_dataset_from_folders",
    "resolve_audio_folder",
    "extract_envelope_features",
    "build_feature_dataframe",
    "compute_spectral_pooling_features",
    "compute_prosody_features",
    "extract_spectral_and_prosody_features",
    "build_spectral_feature_dataframe",
    "balance_speaker_groups",
    "stratified_speaker_train_test_split",
    "get_group_statistics",
    "check_gpu_availability",
    "create_classifier",
    "create_svm_classifier",
    "tune_svm_grid_search",
    "tune_svm_optuna",
    "train_and_evaluate",
    "run_group_cross_validation",
    "analyze_model_stability",
    "run_multiseed_experiment",
    "create_chart",
    "plot_speaker_fragments",
    "plot_speaker_comparison",
    "plot_group_balance",
    "plot_feature_importances",
    "plot_confusion_matrix",
    "plot_spectrogram_and_mfcc_comparison",
    "plot_svm_grid_search",
    "plot_roc_curve",
    "plot_cv_stability",
    "plot_feature_stability",
    "plot_multiseed_stability",
]
