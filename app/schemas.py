from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & Runtime Metadata Schemas (FR-01)
# ---------------------------------------------------------------------------
class EnvironmentInfo(BaseModel):
    python_version: str
    opencv_version: str
    scikit_learn_version: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    models_loaded: List[str]
    pipelines_supported: List[str]
    environment: EnvironmentInfo


# ---------------------------------------------------------------------------
# Single-Image Prediction Schemas (FR-02)
# ---------------------------------------------------------------------------
class PerturbationApplied(BaseModel):
    brightness_shift: float = 0.0
    saturation_shift: float = 0.0
    apply_blur: bool = False


class PredictionMeta(BaseModel):
    pipeline_used: str
    classifier_used: str
    image_resize_shape: List[int] = [384, 128]
    model_cv_macro_f1: float
    total_features_extracted: int = 142
    active_features_used: int = 60
    preprocessing_time_ms: float
    prediction_time_ms: float


class PredictionDetail(BaseModel):
    predicted_class: str
    confidence_score: float
    probabilities: Dict[str, float]


class FeatureSummary(BaseModel):
    leaf_area_ratio: float
    lesion_area_ratio: float
    lesion_density_per_1k_px: float
    estimated_lesion_blobs: int = 0
    dominant_hsv_mean: Optional[List[float]] = None
    glcm_contrast_mean: Optional[float] = None


class VisualizationsBase64(BaseModel):
    step_1_resized_input: str = Field(..., alias="1_resized_input")
    step_2_gray_world: str = Field(..., alias="2_gray_world")
    step_3_clahe_lab: str = Field(..., alias="3_clahe_lab")
    step_4_leaf_mask_morphology: str = Field(..., alias="4_leaf_mask_morphology")
    step_5_lesion_mask: str = Field(..., alias="5_lesion_mask")
    step_6_final_segmented: str = Field(..., alias="6_final_segmented")

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    status: str = "success"
    meta: PredictionMeta
    perturbation_applied: PerturbationApplied
    prediction: PredictionDetail
    feature_summary: FeatureSummary
    visualizations_base64: VisualizationsBase64
    disclaimer: str

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Out-of-Distribution / Non-Leaf Guard Error (FR-03)
# ---------------------------------------------------------------------------
class InvalidLeafErrorResponse(BaseModel):
    status: str = "error"
    error_code: str = "INVALID_LEAF_OBJECT"
    message: str
    leaf_area_ratio: float


# ---------------------------------------------------------------------------
# Pipeline Comparative Stress Test Schemas (FR-04)
# ---------------------------------------------------------------------------
class PipelineComparisonItem(BaseModel):
    clean_class: str
    clean_prediction: Optional[str] = None
    clean_confidence: float
    perturbed_class: str
    perturbed_prediction: Optional[str] = None
    perturbed_confidence: float
    is_consistent: bool
    prediction_consistent: Optional[bool] = None
    confidence_drop: float


class ComparePipelinesResponse(BaseModel):
    status: str = "success"
    classifier_tested: str
    classifier_used: Optional[str] = None
    perturbation_condition: str
    perturbation_tested: Optional[str] = None
    comparisons: Dict[str, PipelineComparisonItem]
    comparison_results: Optional[Dict[str, PipelineComparisonItem]] = None
    scientific_insight: str


# ---------------------------------------------------------------------------
# Experiment Metrics Dashboard Schemas (FR-05)
# ---------------------------------------------------------------------------
class MacroF1Item(BaseModel):
    pipeline: str
    classifier: str
    mean_macro_f1: float
    std_dev: float
    ci_95: List[float]


class PosthocWilcoxon(BaseModel):
    delta_macro_f1: float
    p_holm_corrected: float
    significant: bool


class StatisticalTests(BaseModel):
    friedman_p_value: float
    posthoc_wilcoxon_p1_vs_b0: PosthocWilcoxon


class DatasetSummary(BaseModel):
    total_images: int = 120
    classes: Dict[str, int] = {
        "Bacterial leaf blight": 40,
        "Brown spot": 40,
        "Leaf smut": 40,
    }


class ExperimentMetricsResponse(BaseModel):
    status: str = "success"
    validation_scheme: str = (
        "Repeated Stratified 5-Fold Cross Validation (5 Repeats = 25 Splits)"
    )
    random_seed: int = 42
    dataset_summary: DatasetSummary = DatasetSummary()
    macro_f1_summary: List[MacroF1Item]
    statistical_tests: StatisticalTests
