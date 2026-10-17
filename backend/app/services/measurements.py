import numpy as np
import cv2

def compute_measurements(binary_mask, pixel_spacing=None, is_volumetric=False, voxel_dims=None):
    """
    Computes rigorous geometrical measurements from segmentation masks.
    Explicitly separates pixel-space estimates from calibrated physical metrics.
    """
    if binary_mask is None or binary_mask.sum() == 0:
        return {
            "has_lesion": False,
            "pixel_count": 0,
            "percentage_occupied": 0.0,
            "bounding_box": None,
            "estimated_diameter_px": 0.0,
            "physical_area_cm2": None,
            "physical_volume_ml": None,
            "physical_extent_mm": None,
            "notes": "No lesion detected. Parenchyma area normal."
        }

    mask_bool = (binary_mask > 0)
    pixel_count = int(np.sum(mask_bool))
    total_pixels = int(binary_mask.size)
    percentage = round((pixel_count / total_pixels) * 100.0, 2)

    # 2D Bounding box
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    ymin, ymax = int(row_indices[0]), int(row_indices[-1])
    xmin, xmax = int(col_indices[0]), int(col_indices[-1])
    bbox_width_px = xmax - xmin + 1
    bbox_height_px = ymax - ymin + 1

    # Equivalent circle diameter: D = 2 * sqrt(Area / pi)
    equiv_diameter_px = round(float(2.0 * np.sqrt(pixel_count / np.pi)), 1)

    physical_area_cm2 = None
    physical_extent_mm = None
    physical_volume_ml = None

    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        # e.g., spacing = (0.9375, 0.9375) mm
        dx, dy = float(pixel_spacing[0]), float(pixel_spacing[1])
        area_mm2 = pixel_count * (dx * dy)
        physical_area_cm2 = round(area_mm2 / 100.0, 3)
        physical_extent_mm = {
            "width_mm": round(bbox_width_px * dx, 1),
            "height_mm": round(bbox_height_px * dy, 1),
            "equiv_diameter_mm": round(equiv_diameter_px * np.sqrt(dx * dy), 1)
        }

    if is_volumetric and voxel_dims is not None and len(voxel_dims) == 3:
        # Volumetric NIfTI 3D voxel volume
        vx, vy, vz = voxel_dims
        volume_mm3 = pixel_count * (vx * vy * vz)
        physical_volume_ml = round(volume_mm3 / 1000.0, 2)

    return {
        "has_lesion": True,
        "pixel_count": pixel_count,
        "total_image_pixels": total_pixels,
        "percentage_occupied": percentage,
        "bounding_box_px": {
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
            "width": bbox_width_px, "height": bbox_height_px
        },
        "equivalent_diameter_px": equiv_diameter_px,
        "physical_area_cm2": physical_area_cm2,
        "physical_volume_ml": physical_volume_ml,
        "physical_extent_mm": physical_extent_mm,
        "is_calibrated": pixel_spacing is not None or voxel_dims is not None,
        "disclaimer": "All geometrical measurements are AI-derived computational approximations and require radiologist calibration."
    }
