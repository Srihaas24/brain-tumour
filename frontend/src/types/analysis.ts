export interface ClassProbability {
  class_id: string;
  display_name: string;
  probability: number;
  percentage: number;
}

export interface ClassificationResult {
  predicted_class: string;
  display_name: string;
  confidence: number;
  probabilities: ClassProbability[];
  severity: string;
  description: string;
  clinical_actions: string;
  inference_time_ms: number;
  is_trained_model: boolean;
  status_label: string;
}

export interface SegmentationStats {
  has_lesion: boolean;
  pixel_count: number;
  total_image_pixels: number;
  percentage_occupied: number;
  bounding_box_px?: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
    width: number;
    height: number;
  };
  equivalent_diameter_px: number;
  physical_area_cm2?: number | null;
  physical_volume_ml?: number | null;
  physical_extent_mm?: {
    width_mm: number;
    height_mm: number;
    equiv_diameter_mm: number;
  } | null;
  is_calibrated: boolean;
  disclaimer: string;
}

export interface QualityCheck {
  id: string;
  name: string;
  status: 'PASS' | 'WARNING' | 'FAIL';
  detail: string;
}

export interface QualityAssessmentResult {
  overall_status: 'Acceptable for review' | 'Needs review' | 'Processing failed';
  recommendation: string;
  checks: QualityCheck[];
  warnings: string[];
  sharpness_score: number;
  contrast_score: number;
  estimated_sequence: string;
}


export interface Mesh3DResult {
  has_mesh: boolean;
  vertex_count: number;
  face_count: number;
  vertices: number[][];
  faces: number[][];
  is_genuine_3d: boolean;
  limitation_note: string;
  anatomical_location?: string;
  hemisphere?: string;
  centroid_3d?: number[];
}

export interface HumanReviewRecommendation {
  review_status: string;
  reasons: string[];
  requires_human_attention: boolean;
  disclaimer: string;
}

export interface AnalysisResponse {
  result_id: string;
  filename: string;
  model_used: string;
  is_volumetric: boolean;
  metadata: {
    is_nifti?: boolean;
    dimensions?: string;
    total_slices?: number;
    current_slice?: number;
    voxel_spacing?: number[] | null;
  };
  original_image_base64: string;
  mask_base64: string;
  overlay_base64: string;
  transparent_overlay_base64?: string;
  classification: ClassificationResult;
  segmentation: SegmentationStats;
  quality: QualityAssessmentResult;
  mesh_3d: Mesh3DResult;
  human_review: HumanReviewRecommendation;
  total_pipeline_time_ms: number;
}

export interface BenchmarkModel {
  name: string;
  parameters: string;
  param_count: number;
  latency_ms: number;
  memory_mb: number;
  val_dice_score: number;
  val_iou: number;
  compression_ratio: string;
  intended_deployment: string;
}

export interface BenchmarkResponse {
  models: BenchmarkModel[];
  device_used: string;
  speedup: number;
}
