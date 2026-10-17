import os
import io
import base64
import numpy as np
import cv2
from PIL import Image

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def decode_image_bytes(file_bytes, filename="image.jpg", slice_idx=None):
    """
    Decodes uploaded raw bytes into RGB NumPy image, handling standard images and NIfTI volumes.
    Returns:
        image_np: (H, W, 3) float32 in [0, 255]
        is_volumetric: bool
        metadata: dict
        raw_volume: optional (D, H, W) numpy array
    """
    fname_lower = filename.lower()
    
    # 1. Handle NIfTI volumes
    if fname_lower.endswith(('.nii', '.nii.gz')):
        if not HAS_NIBABEL:
            raise RuntimeError("nibabel is not installed on the system to process NIfTI files.")
        
        # Save to temp buffer for nibabel
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            nii = nib.load(tmp_path)
            data = nii.get_fdata()
            zooms = nii.header.get_zooms()
            os.remove(tmp_path)

            # Squeeze extra dims
            data = np.squeeze(data)
            if len(data.shape) == 3:
                # Shape (X, Y, Z), choose axial slices along Z (dim 2)
                d_slices = data.shape[2]
                target_slice = slice_idx if (slice_idx is not None and 0 <= slice_idx < d_slices) else d_slices // 2
                slice_2d = data[:, :, target_slice]

                # Normalize to 0-255
                slice_min, slice_max = slice_2d.min(), slice_2d.max()
                if slice_max > slice_min:
                    slice_norm = ((slice_2d - slice_min) / (slice_max - slice_min) * 255.0).astype(np.uint8)
                else:
                    slice_norm = np.zeros_like(slice_2d, dtype=np.uint8)

                img_rgb = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2RGB)
                metadata = {
                    "is_nifti": True,
                    "volume_shape": list(data.shape),
                    "total_slices": d_slices,
                    "current_slice": target_slice,
                    "voxel_spacing": [float(z) for z in zooms[:3]],
                    "dimensions": f"{data.shape[0]}x{data.shape[1]}x{data.shape[2]}"
                }
                return img_rgb, True, metadata, data
            else:
                slice_norm = ((data - data.min()) / (data.max() - data.min() + 1e-6) * 255.0).astype(np.uint8)
                img_rgb = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2RGB)
                return img_rgb, False, {"is_nifti": True, "dimensions": f"{data.shape[0]}x{data.shape[1]}"}, None
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError(f"Failed to parse NIfTI volume: {e}")

    # 2. Standard 2D image (PNG, JPG, JPEG)
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        img_np = np.array(pil_img)
        metadata = {
            "is_nifti": False,
            "dimensions": f"{img_np.shape[1]}x{img_np.shape[0]}",
            "channels": 3,
            "total_slices": 1,
            "current_slice": 0,
            "voxel_spacing": None
        }
        return img_np, False, metadata, None
    except Exception as e:
        raise ValueError(f"Failed to decode image raster: {e}")


def make_clinical_overlay(original_mri, binary_mask, alpha=0.45):
    """
    Renders a clinical translucent crimson overlay with a high-contrast gold border contour.
    """
    overlay = original_mri.copy().astype(np.float32) / 255.0
    mask_bool = (binary_mask > 0)

    if np.any(mask_bool):
        # Crimson tint over tumor
        red_tint = np.array([1.0, 0.15, 0.15], dtype=np.float32)
        for c in range(3):
            overlay[mask_bool, c] = (1.0 - alpha) * overlay[mask_bool, c] + alpha * red_tint[c]

        # Draw gold contour
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_uint8 = (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)
        cv2.drawContours(overlay_uint8, contours, -1, (255, 215, 0), 2)
        return overlay_uint8

    return (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)


def make_transparent_mask(binary_mask):
    """
    Renders an RGBA transparent image layer where non-tumor background is completely
    transparent (alpha=0), tumor tissue is translucent crimson [244, 63, 94, 190],
    and tumor boundaries are demarcated with an opaque gold contour [255, 215, 0, 255].
    """
    h, w = binary_mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bool = (binary_mask > 0)

    if np.any(mask_bool):
        # Crimson tint for tumor core
        rgba[mask_bool] = [244, 63, 94, 190]

        # Draw gold contour on contour margin
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgba, contours, -1, (255, 215, 0, 255), 2)

    return rgba


def numpy_to_base64(img_np, format="PNG"):
    buf = io.BytesIO()
    if len(img_np.shape) == 2:
        Image.fromarray(img_np.astype(np.uint8)).save(buf, format=format)
    else:
        Image.fromarray(img_np.astype(np.uint8)).save(buf, format=format)
    return f"data:image/{format.lower()};base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

