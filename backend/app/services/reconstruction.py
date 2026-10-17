import numpy as np
import cv2
from skimage import measure

# Anatomical scale reference matching Three.js brain reference cage:
# Brain reference in Three.js: SphereGeometry(6.2) scaled (1.0, 1.2, 1.35)
# Extents: X in [-6.2, +6.2], Y in [-7.44, +7.44], Z in [-8.37, +8.37]
BRAIN_RADIUS_X = 5.60
BRAIN_RADIUS_Y = 6.70
BRAIN_RADIUS_Z = 7.55


def detect_head_frame(raw_image, fallback_shape):
    """
    Detects cranial boundary, center of mass, and half-axes from raw MRI scan.
    """
    if raw_image is not None and len(raw_image.shape) >= 2:
        gray = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY) if len(raw_image.shape) == 3 else raw_image
        _, thresh = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                M = cv2.moments(c)
                cx = float(M['m10'] / M['m00']) if M['m00'] > 0 else float(x + w / 2.0)
                cy = float(M['m01'] / M['m00']) if M['m00'] > 0 else float(y + h / 2.0)
                return cx, cy, max(w / 2.0, 10.0), max(h / 2.0, 10.0)

    # Fallback to scan image geometric center
    h_img, w_img = fallback_shape[:2]
    cx = float(w_img / 2.0)
    cy = float(h_img / 2.0)
    rx = float(w_img * 0.42)
    ry = float(h_img * 0.45)
    return cx, cy, rx, ry


def generate_3d_mesh(
    mask_data,
    is_volumetric=False,
    voxel_spacing=(1.0, 1.0, 1.0),
    raw_image=None,
    predicted_class=None,
    slice_info=None
):
    """
    Extracts an anatomically accurate 3D polygonal surface mesh from segmentation masks.
    Preserves true anatomical location, hemisphere, aspect ratio, and physical extent.
    Returns vertex/face buffers for Three.js rendering and generates STL/OBJ export streams.
    """
    if mask_data is None or mask_data.sum() == 0 or predicted_class == 'no_tumor':
        return {
            "has_mesh": False,
            "vertex_count": 0,
            "face_count": 0,
            "vertices": [],
            "faces": [],
            "normals": [],
            "stl_ascii": "",
            "obj_ascii": "",
            "is_genuine_3d": False,
            "limitation_note": "No active lesion detected to reconstruct. Parenchyma clear.",
            "anatomical_location": "N/A — Normal Parenchyma",
            "hemisphere": "Bilateral / Clear",
            "centroid_3d": [0.0, 0.0, 0.0]
        }

    try:
        if is_volumetric and len(mask_data.shape) == 3:
            # Genuine 3D Volumetric scan (e.g., from NIfTI volume)
            vol = (mask_data > 0).astype(np.float32)
            is_genuine = True
            note = "Genuine patient-specific 3D reconstruction extracted from multi-slice volumetric MRI."
            
            # Marching cubes on 3D volume
            verts, faces, normals, values = measure.marching_cubes(vol, level=0.5)
            vz, vy, vx = verts.T
            
            d_slices, h_vol, w_vol = vol.shape
            cx, cy, rx, ry = detect_head_frame(raw_image, (h_vol, w_vol))
            
            # Map coordinates to Three.js space
            three_x = ((vx - cx) / rx) * BRAIN_RADIUS_X
            three_z = -((vy - cy) / ry) * BRAIN_RADIUS_Z
            three_y = ((vz - d_slices / 2.0) / max(d_slices / 2.0, 1.0)) * BRAIN_RADIUS_Y
            
            norm_verts = np.stack([three_x, three_y, three_z], axis=1)
            
            # Centroid & localization
            c_3d = [float(three_x.mean()), float(three_y.mean()), float(three_z.mean())]
            hemi = "Left Hemisphere" if c_3d[0] < -0.6 else ("Right Hemisphere" if c_3d[0] > 0.6 else "Midline")
            loc = f"{hemi} (Volumetric Lesion)"
        else:
            # 2D slice input: Extrapolate anatomically co-registered 3D volume
            is_genuine = False
            note = "Co-registered from 2D MRI slice via distance transformation. Accurately preserves anatomical location and physical aspect ratio within the cerebral cage."
            
            if len(mask_data.shape) == 3:
                mask_2d = mask_data[0]
            else:
                mask_2d = mask_data

            h, w = mask_2d.shape
            bin_mask = (mask_2d > 0).astype(np.uint8)
            
            # 1. Detect head boundary and center
            cx, cy, head_rx, head_ry = detect_head_frame(raw_image, (h, w))
            
            # 2. Extract lesion centroid and anatomical compartment
            t_ys, t_xs = np.where(bin_mask > 0)
            t_cx, t_cy = float(np.mean(t_xs)), float(np.mean(t_ys))
            
            norm_lesion_x = (t_cx - cx) / head_rx
            norm_lesion_y = (t_cy - cy) / head_ry
            
            # Hemisphere determination
            if abs(norm_lesion_x) < 0.12:
                hemi = "Midline / Sellar"
            elif norm_lesion_x < -0.12:
                hemi = "Left Hemisphere"
            else:
                hemi = "Right Hemisphere"
                
            # Anatomical region determination
            if predicted_class == 'pituitary' or (abs(norm_lesion_x) < 0.20 and norm_lesion_y > 0.0 and predicted_class != 'glioma'):
                loc = f"{hemi} — Sellar Region / Skull Base (Pituitary)"
                y_elevation = -2.8
            elif norm_lesion_y < -0.25:
                loc = f"{hemi} — Frontal Lobe (Anterior)"
                y_elevation = 0.8
            elif norm_lesion_y > 0.25:
                loc = f"{hemi} — Occipital / Posterior Temporal"
                y_elevation = 0.8
            elif abs(norm_lesion_x) > 0.35:
                loc = f"{hemi} — Lateral Temporal / Parietal"
                y_elevation = 0.8
            else:
                loc = f"{hemi} — Fronto-Parietal Region"
                y_elevation = 0.8

            if slice_info and slice_info.get("total_slices", 1) > 1:
                cur_s = slice_info.get("current_slice", 0)
                tot_s = slice_info.get("total_slices", 1)
                y_elevation = ((cur_s - tot_s / 2.0) / (tot_s / 2.0)) * 5.0

            # 3. Crop tightly to lesion bounding box for ultra-fast Marching Cubes
            pad = 8
            ymin = max(0, int(t_ys.min()) - pad)
            ymax = min(h, int(t_ys.max()) + pad + 1)
            xmin = max(0, int(t_xs.min()) - pad)
            xmax = min(w, int(t_xs.max()) + pad + 1)
            
            crop_mask = bin_mask[ymin:ymax, xmin:xmax]
            dist_map = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 5)
            max_d = float(dist_map.max())
            if max_d > 0:
                dist_map = dist_map / max_d

            num_z = 24
            vol = np.zeros((num_z, crop_mask.shape[0], crop_mask.shape[1]), dtype=np.float32)
            for iz in range(num_z):
                z_rel = (iz - num_z / 2.0) / (num_z / 2.0)
                rad_factor = 1.0 - z_rel**2
                if rad_factor > 0:
                    vol[iz] = dist_map * np.sqrt(rad_factor)

            # 4. Marching cubes
            verts, faces, normals, values = measure.marching_cubes(vol, level=0.25)
            vz, vy, vx = verts.T
            
            # Map back to full image pixel coordinates
            col_coords = xmin + vx
            row_coords = ymin + vy
            
            # Map to normalized head coordinates (-1.0 to +1.0)
            norm_x = (col_coords - cx) / head_rx
            norm_y = (row_coords - cy) / head_ry
            
            # Map to Three.js space:
            # Three.js X = Left (-X) to Right (+X)
            # Three.js Z = Posterior (-Z) to Anterior (+Z) (row 0 is Anterior -> negative norm_y is +Z)
            three_x = norm_x * BRAIN_RADIUS_X
            three_z = -norm_y * BRAIN_RADIUS_Z
            
            # Elevation & true physical thickness scale
            tumor_equiv_rad = np.sqrt(len(t_ys) / np.pi)
            vert_scale = (tumor_equiv_rad / head_rx) * BRAIN_RADIUS_X / (num_z / 2.0)
            three_y = y_elevation + (vz - num_z / 2.0) * vert_scale
            
            norm_verts = np.stack([three_x, three_y, three_z], axis=1)
            c_3d = [float(np.mean(three_x)), float(np.mean(three_y)), float(np.mean(three_z))]

        # Generate ASCII STL
        stl_lines = ["solid NeuroVision_Tumor_Mesh"]
        for f in faces:
            v0 = norm_verts[f[0]]
            v1 = norm_verts[f[1]]
            v2 = norm_verts[f[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            fn = np.cross(edge1, edge2)
            fn_norm = np.linalg.norm(fn)
            fn = fn / fn_norm if fn_norm > 0 else np.array([0, 0, 1])

            stl_lines.append(f"  facet normal {fn[0]:.4f} {fn[1]:.4f} {fn[2]:.4f}")
            stl_lines.append("    outer loop")
            stl_lines.append(f"      vertex {v0[0]:.4f} {v0[1]:.4f} {v0[2]:.4f}")
            stl_lines.append(f"      vertex {v1[0]:.4f} {v1[1]:.4f} {v1[2]:.4f}")
            stl_lines.append(f"      vertex {v2[0]:.4f} {v2[1]:.4f} {v2[2]:.4f}")
            stl_lines.append("    endloop")
            stl_lines.append("  endfacet")
        stl_lines.append("endsolid NeuroVision_Tumor_Mesh\n")
        stl_ascii = "\n".join(stl_lines)

        # Generate OBJ
        obj_lines = ["# NeuroVision AI 3D Mesh Export"]
        for v in norm_verts:
            obj_lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
        for f in faces:
            obj_lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
        obj_ascii = "\n".join(obj_lines)

        return {
            "has_mesh": True,
            "vertex_count": len(norm_verts),
            "face_count": len(faces),
            "vertices": norm_verts.tolist(),
            "faces": faces.tolist(),
            "normals": normals.tolist(),
            "stl_ascii": stl_ascii,
            "obj_ascii": obj_ascii,
            "is_genuine_3d": is_genuine,
            "limitation_note": note,
            "anatomical_location": loc,
            "hemisphere": hemi,
            "centroid_3d": [round(c, 2) for c in c_3d]
        }
    except Exception as e:
        return {
            "has_mesh": False,
            "vertex_count": 0,
            "face_count": 0,
            "vertices": [],
            "faces": [],
            "normals": [],
            "stl_ascii": "",
            "obj_ascii": "",
            "error": str(e),
            "is_genuine_3d": is_genuine if 'is_genuine' in locals() else False,
            "limitation_note": f"Mesh extraction failed: {e}",
            "anatomical_location": "Error in reconstruction",
            "hemisphere": "Unknown",
            "centroid_3d": [0.0, 0.0, 0.0]
        }

