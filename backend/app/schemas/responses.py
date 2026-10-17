from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ClassProbability(BaseModel):
    class_id: str
    display_name: str
    probability: float
    percentage: float

class ClassificationResult(BaseModel):
    predicted_class: str
    display_name: str
    confidence: float
    probabilities: List[ClassProbability]
    severity: str
    description: str
    clinical_actions: str
    inference_time_ms: float
    is_trained_model: bool
    status_label: str

class SegmentationStats(BaseModel):
    has_lesion: bool
    pixel_count: int
    percentage_occupied: float
    bounding_box_px: Optional[Dict[str, int]]
    equivalent_diameter_px: float
    physical_area_cm2: Optional[float]
    physical_volume_ml: Optional[float]
    physical_extent_mm: Optional[Dict[str, float]]
    is_calibrated: bool
    disclaimer: str

class QualityCheck(BaseModel):
    id: str
    name: str
    status: str
    detail: str

class QualityAssessmentResult(BaseModel):
    overall_status: str
    recommendation: str
    checks: List[QualityCheck]
    warnings: List[str]
    sharpness_score: float
    contrast_score: float
    estimated_sequence: str

class UncertaintyResult(BaseModel):
    boundary_uncertainty: float
    overall_uncertainty_score: float
    num_stochastic_passes: int
    review_recommended: bool
    heatmap_base64: str
    overlay_base64: Optional[str] = None

class Mesh3DResult(BaseModel):
    has_mesh: bool
    vertex_count: int
    face_count: int
    vertices: List[List[float]]
    faces: List[List[int]]
    is_genuine_3d: bool
    limitation_note: str
    anatomical_location: Optional[str] = None
    hemisphere: Optional[str] = None
    centroid_3d: Optional[List[float]] = None

class HumanReviewRecommendation(BaseModel):
    review_status: str
    reasons: List[str]
    requires_human_attention: bool
    disclaimer: str

class AnalysisResponse(BaseModel):
    result_id: str
    filename: str
    model_used: str
    is_volumetric: bool
    metadata: Dict[str, Any]
    original_image_base64: str
    mask_base64: str
    overlay_base64: str
    transparent_overlay_base64: Optional[str] = None
    classification: ClassificationResult
    segmentation: SegmentationStats
    uncertainty: UncertaintyResult
    quality: QualityAssessmentResult
    mesh_3d: Mesh3DResult
    human_review: HumanReviewRecommendation
    total_pipeline_time_ms: float
