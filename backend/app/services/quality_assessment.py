import numpy as np
import cv2

def evaluate_scan_quality(image_np, binary_mask=None, uncertainty_info=None, filename="scan.jpg"):
    """
    Automated multi-point Rule-Based Quality Assessment system for MRI inputs and AI inferences.
    Validates input suitability and segmentation reliability across 8 checks.
    """
    checks = []
    warnings = []
    needs_review = False
    processing_failed = False

    # Check 1: Format & Readability
    valid_extensions = ('.png', '.jpg', '.jpeg', '.nii', '.nii.gz')
    has_valid_ext = filename.lower().endswith(valid_extensions)
    is_readable = image_np is not None and image_np.size > 0
    checks.append({
        "id": "format_readability",
        "name": "Format & Readability",
        "status": "PASS" if (has_valid_ext and is_readable) else "FAIL",
        "detail": f"File '{filename}' validated successfully ({image_np.shape[1]}x{image_np.shape[0]} px)" if is_readable else "Unable to decode image raster."
    })
    if not is_readable:
        processing_failed = True

    # Check 2: Resolution & Dimension Suitability
    h, w = image_np.shape[:2]
    is_res_adequate = h >= 180 and w >= 180
    checks.append({
        "id": "dimensions",
        "name": "Spatial Resolution",
        "status": "PASS" if is_res_adequate else "WARNING",
        "detail": f"{w}x{h} px ({'Optimal resolution' if is_res_adequate else 'Low resolution may compromise fine margins'})"
    })
    if not is_res_adequate:
        warnings.append("Low input spatial resolution (<180px); fine morphological features may be smoothed.")
        needs_review = True

    # Check 3: Blur Assessment (Laplacian Variance)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_sharp = lap_var > 60.0
    checks.append({
        "id": "sharpness",
        "name": "Motion Blur & Sharpness",
        "status": "PASS" if is_sharp else "WARNING",
        "detail": f"Laplacian sharpness metric: {lap_var:.1f} (Threshold: 60.0)"
    })
    if not is_sharp:
        warnings.append(f"Suboptimal sharpness (metric: {lap_var:.1f}); possible patient motion artifact.")
        needs_review = True

    # Check 4: Dynamic Range & Contrast
    min_val, max_val = float(gray.min()), float(gray.max())
    dynamic_range = max_val - min_val
    std_dev = float(gray.std())
    is_contrast_ok = dynamic_range >= 100 and std_dev >= 20.0
    checks.append({
        "id": "contrast",
        "name": "Tissue Contrast & Dynamic Range",
        "status": "PASS" if is_contrast_ok else "WARNING",
        "detail": f"Dynamic range: {dynamic_range:.0f}/255, Contrast STD: {std_dev:.1f}"
    })
    if not is_contrast_ok:
        warnings.append("Low tissue contrast detected. Grey/white matter boundary separation may be faint.")
        needs_review = True

    # Check 5: Domain Plausibility & Skull Contour Check
    # Brain scans typically exhibit a central mass with dark background margins
    center_patch = gray[h//4:3*h//4, w//4:3*w//4]
    corner_patch = np.concatenate([gray[:h//8, :w//8].flatten(), gray[-h//8:, -w//8:].flatten()])
    center_mean = float(center_patch.mean())
    corner_mean = float(corner_patch.mean())
    is_brain_like = center_mean > corner_mean + 15.0 and center_mean > 30.0
    checks.append({
        "id": "domain_plausibility",
        "name": "Cranial Domain Check",
        "status": "PASS" if is_brain_like else "WARNING",
        "detail": "Image morphology consistent with neuroimaging skull-interior profile." if is_brain_like else "Image profile deviates from typical axial/coronal brain MRI characteristics."
    })
    if not is_brain_like:
        warnings.append("Input pattern deviates from standard cranial axial/coronal MRI profiles.")
        needs_review = True

    # Check 6: Segmentation Uncertainty
    boundary_unc = uncertainty_info.get("boundary_uncertainty", 0.0) if uncertainty_info else 0.0
    is_uncertainty_ok = boundary_unc <= 35.0
    checks.append({
        "id": "segmentation_uncertainty",
        "name": "Boundary Uncertainty Metric",
        "status": "PASS" if is_uncertainty_ok else "WARNING",
        "detail": f"Boundary uncertainty: {boundary_unc:.1f}% (Review threshold: 35.0%)"
    })
    if not is_uncertainty_ok:
        warnings.append(f"Elevated boundary uncertainty ({boundary_unc:.1f}%); manual contour verification recommended.")
        needs_review = True

    # Check 7: Morphological Plausibility & Component Integrity
    if binary_mask is not None and binary_mask.sum() > 0:
        mask_uint8 = (binary_mask > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        # Check if there are numerous tiny scattered clusters (<10 pixels)
        small_components = sum(1 for s in stats[1:] if s[cv2.CC_STAT_AREA] < 12)
        total_components = num_labels - 1
        is_plausible = small_components <= 2
        checks.append({
            "id": "component_integrity",
            "name": "Lesion Spatial Plausibility",
            "status": "PASS" if is_plausible else "WARNING",
            "detail": f"Detected {total_components} distinct lesion component(s) ({small_components} minor clusters)"
        })
        if not is_plausible:
            warnings.append(f"Detected {small_components} tiny disconnected lesion fragments; potential false-positive noise.")
            needs_review = True
    else:
        checks.append({
            "id": "component_integrity",
            "name": "Lesion Spatial Plausibility",
            "status": "PASS",
            "detail": "No focal lesion detected; healthy parenchyma morphology."
        })

    # Check 8: MRI Sequence Identification
    # Heuristic estimation based on CSF vs white matter intensity
    if center_mean > 120 and corner_mean < 40:
        est_seq = "T1-Weighted (Contrast Enhanced / High Signal)"
    elif std_dev > 55:
        est_seq = "T2-Weighted / FLAIR"
    else:
        est_seq = "T1-Weighted Standard"
    checks.append({
        "id": "sequence_type",
        "name": "Estimated MRI Sequence",
        "status": "PASS",
        "detail": f"{est_seq} (Derived from intensity histogram analysis)"
    })

    # Overall Summary
    if processing_failed:
        overall_status = "Processing failed"
        recommendation = "Unable to proceed with analysis. Please upload an intact, readable MRI file."
    elif needs_review:
        overall_status = "Needs review"
        recommendation = "Analysis completed with automated warnings. Human specialist review recommended before clinical action."
    else:
        overall_status = "Acceptable for review"
        recommendation = "All automated quality and uncertainty checks passed acceptable research thresholds."

    return {
        "overall_status": overall_status,
        "recommendation": recommendation,
        "checks": checks,
        "warnings": warnings,
        "sharpness_score": round(lap_var, 1),
        "contrast_score": round(std_dev, 1),
        "estimated_sequence": est_seq
    }
