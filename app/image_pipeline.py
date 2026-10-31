import base64
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

from app.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    MAX_FILE_SIZE_BYTES,
    MIN_LEAF_AREA_RATIO,
)


class InvalidImageError(ValueError):
    """Raised when an uploaded file is not a valid JPEG or PNG image."""
    pass


class NonLeafImageError(ValueError):
    """Raised when an image contains insufficient leaf area (leaf_area_ratio < threshold)."""
    def __init__(self, message: str, leaf_area_ratio: float):
        super().__init__(message)
        self.leaf_area_ratio = leaf_area_ratio


def validate_image_file(file_bytes: bytes, filename: str = "") -> None:
    """
    Validates file size and magic bytes to ensure file integrity.
    Accepts only JPG/JPEG and PNG files up to 10 MB.
    """
    if len(file_bytes) == 0:
        raise InvalidImageError("File gambar kosong.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise InvalidImageError(
            f"Ukuran file melebihi batas maksimum 10 MB (terdeteksi: {len(file_bytes)/(1024*1024):.2f} MB)."
        )

    # Magic Bytes check
    is_jpeg = file_bytes.startswith(b"\xff\xd8\xff")
    is_png = file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpeg or is_png):
        raise InvalidImageError(
            "Format berkas tidak valid. Harap unggah file foto berformat .jpg, .jpeg, atau .png."
        )


def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes into an OpenCV BGR numpy array."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise InvalidImageError("Gagal membaca data citra. Pastikan berkas gambar tidak rusak.")
    return img_bgr


def encode_image_base64(img: np.ndarray, format_type: str = "jpeg") -> str:
    """Encodes an image array into a Data URI Base64 string."""
    if format_type.lower() == "png":
        success, buffer = cv2.imencode(".png", img)
        mime = "image/png"
    else:
        success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        mime = "image/jpeg"

    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"


def apply_perturbations(
    img_bgr: np.ndarray,
    brightness_shift: float = 0.0,
    saturation_shift: float = 0.0,
    apply_blur: bool = False,
) -> np.ndarray:
    """
    Simulates lighting and capture perturbations on-the-fly:
    - brightness_shift: [-0.20, +0.20]
    - saturation_shift: [-0.15, +0.15]
    - apply_blur: Gaussian Blur
    """
    res = img_bgr.astype(np.float32)

    # 1. Brightness shift
    if abs(brightness_shift) > 1e-4:
        shift_val = np.clip(brightness_shift, -0.20, 0.20) * 255.0
        res = np.clip(res + shift_val, 0, 255)

    res_uint8 = res.astype(np.uint8)

    # 2. Saturation shift in HSV
    if abs(saturation_shift) > 1e-4:
        hsv = cv2.cvtColor(res_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_factor = 1.0 + np.clip(saturation_shift, -0.15, 0.15)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        res_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Gaussian Blur
    if apply_blur:
        res_uint8 = cv2.GaussianBlur(res_uint8, (5, 5), 1.2)

    return res_uint8


def apply_gray_world(img_bgr: np.ndarray) -> np.ndarray:
    """Gray World color constancy algorithm."""
    img_f = img_bgr.astype(np.float32)
    b_mean = float(np.mean(img_f[:, :, 0]))
    g_mean = float(np.mean(img_f[:, :, 1]))
    r_mean = float(np.mean(img_f[:, :, 2]))

    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    img_f[:, :, 0] *= gray_mean / (b_mean + 1e-5)
    img_f[:, :, 1] *= gray_mean / (g_mean + 1e-5)
    img_f[:, :, 2] *= gray_mean / (r_mean + 1e-5)

    return np.clip(img_f, 0, 255).astype(np.uint8)


def apply_clahe_lab(img_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Applies CLAHE enhancement on the L-channel of the Lab color space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def preprocess_image(
    img_bgr: np.ndarray, pipeline_mode: str = "P1"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes the designated image processing pipeline scenario:
    - B0: Baseline 0 (Resize only)
    - B1: Baseline 1 (Resize -> Gray World)
    - B2: Baseline 2 (Resize -> CLAHE Lab)
    - P1: Proposed 1 (Resize -> Gray World -> CLAHE Lab)

    Returns:
    (processed_img, gw_visual, clahe_visual)
    """
    # Standard resize to (width=128, height=384)
    resized = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

    if pipeline_mode == "B0":
        return resized, resized, resized
    elif pipeline_mode == "B1":
        gw = apply_gray_world(resized)
        return gw, gw, gw
    elif pipeline_mode == "B2":
        clahe = apply_clahe_lab(resized)
        return clahe, resized, clahe
    else:  # "P1"
        gw = apply_gray_world(resized)
        clahe = apply_clahe_lab(gw)
        return clahe, gw, clahe


def segment_leaf_and_lesions(
    img_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """
    Segments the rice leaf object from background and detects lesion blobs.

    Returns:
    - leaf_mask: Binary mask of leaf
    - lesion_mask: Binary mask of lesions
    - final_segmented: Image displaying segmented leaf
    - leaf_area_ratio: Ratio of leaf pixels to total image pixels
    - lesion_area_ratio: Ratio of lesion pixels to leaf pixels
    - lesion_blob_count: Number of distinct lesion blobs
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold on Saturation (leaf has color, background is white/light)
    _, mask_sat = cv2.threshold(hsv[:, :, 1], 25, 255, cv2.THRESH_BINARY)
    # Otsu threshold on inverted grayscale
    _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    leaf_mask_raw = cv2.bitwise_or(mask_sat, mask_otsu)

    # Morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    total_pixels = leaf_mask.size
    leaf_pixels = int(np.count_nonzero(leaf_mask))
    leaf_area_ratio = float(leaf_pixels / max(1, total_pixels))

    # Guard check: if leaf area ratio < threshold, raise NonLeafImageError
    if leaf_area_ratio < MIN_LEAF_AREA_RATIO:
        raise NonLeafImageError(
            f"Area daun padi tidak terdeteksi secara memadai (rasio daun: {leaf_area_ratio*100:.1f}%). "
            f"Pastikan fokus kamera mengarah pada daun padi.",
            leaf_area_ratio=round(leaf_area_ratio, 4),
        )

    # Lesion Segmentation inside leaf
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0]

    # Masked L channel for leaf pixels
    leaf_l = l_channel[leaf_mask > 0]
    l_mean = float(np.mean(leaf_l)) if len(leaf_l) > 0 else 128.0
    l_std = float(np.std(leaf_l)) if len(leaf_l) > 0 else 20.0

    # Lesions are darker or distinct spots on the leaf
    lesion_threshold = max(30, int(l_mean - 0.75 * l_std))
    _, lesion_mask_raw = cv2.threshold(l_channel, lesion_threshold, 255, cv2.THRESH_BINARY_INV)
    lesion_mask = cv2.bitwise_and(lesion_mask_raw, leaf_mask)

    # Remove microscopic noise
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, k_small)

    lesion_pixels = int(np.count_nonzero(lesion_mask))
    lesion_area_ratio = float(lesion_pixels / max(1, leaf_pixels))

    # Count lesion blobs
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(lesion_mask, connectivity=8)
    lesion_blob_count = max(0, num_labels - 1)  # exclude background

    # Final segmented visualization
    final_segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=leaf_mask)

    return (
        leaf_mask,
        lesion_mask,
        final_segmented,
        leaf_area_ratio,
        lesion_area_ratio,
        lesion_blob_count,
    )


def extract_features(
    img_bgr: np.ndarray,
    leaf_mask: np.ndarray,
    lesion_mask: np.ndarray,
    leaf_area_ratio: float,
    lesion_area_ratio: float,
    lesion_blob_count: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts the exact 140-dimensional feature vector matching the model bundle:
    - 0..37 (38): Color Histograms (H: 18 bins, S: 10 bins, V: 10 bins)
    - 38..67 (30): Color Moments for (H, S, V, L, a, b) [mean, std, skew, p25, p75]
    - 68..127 (60): GLCM properties (contrast, dissimilarity, homogeneity, energy, correlation)
    - 128..139 (12): Geometric & lesion morphology features

    Returns:
    - feature_vector: 1D numpy array of shape (140,)
    - feature_summary: Summary dictionary for the API response
    """
    features: List[float] = []

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mask_bool = leaf_mask > 0

    # -----------------------------------------------------------------------
    # Part 1: Color Histograms (38 features)
    # H: 18 bins [0, 180]
    # S: 10 bins [0, 256]
    # V: 10 bins [0, 256]
    # -----------------------------------------------------------------------
    h_pixels = hsv[:, :, 0][mask_bool]
    s_pixels = hsv[:, :, 1][mask_bool]
    v_pixels = hsv[:, :, 2][mask_bool]

    if len(h_pixels) == 0:
        h_pixels = hsv[:, :, 0].flatten()
        s_pixels = hsv[:, :, 1].flatten()
        v_pixels = hsv[:, :, 2].flatten()

    h_hist, _ = np.histogram(h_pixels, bins=18, range=(0, 180))
    s_hist, _ = np.histogram(s_pixels, bins=10, range=(0, 256))
    v_hist, _ = np.histogram(v_pixels, bins=10, range=(0, 256))

    h_norm = h_hist.astype(np.float32) / (np.sum(h_hist) + 1e-6)
    s_norm = s_hist.astype(np.float32) / (np.sum(s_hist) + 1e-6)
    v_norm = v_hist.astype(np.float32) / (np.sum(v_hist) + 1e-6)

    features.extend(h_norm.tolist())  # 18
    features.extend(s_norm.tolist())  # 10
    features.extend(v_norm.tolist())  # 10
    # Total Part 1 = 38

    # -----------------------------------------------------------------------
    # Part 2: Color Moments for 6 Channels (30 features)
    # Channels: H, S, V, L, a, b
    # Statistics: mean, std, skewness, 25th percentile, 75th percentile
    # -----------------------------------------------------------------------
    l_pixels = lab[:, :, 0][mask_bool]
    a_pixels = lab[:, :, 1][mask_bool]
    b_pixels = lab[:, :, 2][mask_bool]

    channels = [h_pixels, s_pixels, v_pixels, l_pixels, a_pixels, b_pixels]
    for ch in channels:
        if len(ch) > 0:
            m = float(np.mean(ch))
            s = float(np.std(ch))
            sk = float(skew(ch)) if len(ch) > 2 else 0.0
            p25 = float(np.percentile(ch, 25))
            p75 = float(np.percentile(ch, 75))
        else:
            m, s, sk, p25, p75 = 0.0, 0.0, 0.0, 0.0, 0.0
        features.extend([m, s, sk, p25, p75])
    # Total Part 2 = 30 (38 + 30 = 68)

    # -----------------------------------------------------------------------
    # Part 3: GLCM Texture Features (60 features)
    # Quantization to 64 levels, 3 distances [1, 2, 4], 4 angles [0, pi/4, pi/2, 3pi/4]
    # Properties: contrast, dissimilarity, homogeneity, energy, correlation
    # -----------------------------------------------------------------------
    q_gray = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(
        q_gray,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )

    glcm_contrast = graycoprops(glcm, "contrast").ravel().tolist()
    glcm_dissim = graycoprops(glcm, "dissimilarity").ravel().tolist()
    glcm_homog = graycoprops(glcm, "homogeneity").ravel().tolist()
    glcm_energy = graycoprops(glcm, "energy").ravel().tolist()
    glcm_corr = graycoprops(glcm, "correlation").ravel().tolist()

    features.extend(glcm_contrast)  # 12
    features.extend(glcm_dissim)  # 12
    features.extend(glcm_homog)  # 12
    features.extend(glcm_energy)  # 12
    features.extend(glcm_corr)  # 12
    # Total Part 3 = 60 (68 + 60 = 128)

    # -----------------------------------------------------------------------
    # Part 4: Shape & Lesion Morphology Features (12 features)
    # -----------------------------------------------------------------------
    lesion_mask_bool = lesion_mask > 0
    if np.any(lesion_mask_bool):
        lesion_b = float(np.mean(img_bgr[:, :, 0][lesion_mask_bool]))
        lesion_g = float(np.mean(img_bgr[:, :, 1][lesion_mask_bool]))
        lesion_r = float(np.mean(img_bgr[:, :, 2][lesion_mask_bool]))
        lesion_h = float(np.mean(hsv[:, :, 0][lesion_mask_bool]))
        lesion_s = float(np.mean(hsv[:, :, 1][lesion_mask_bool]))
        lesion_v = float(np.mean(hsv[:, :, 2][lesion_mask_bool]))
        lesion_l = float(np.mean(lab[:, :, 0][lesion_mask_bool]))
    else:
        lesion_b, lesion_g, lesion_r = 82.9, 48.4, 74.6
        lesion_h, lesion_s, lesion_v, lesion_l = 164.2, 145.4, 133.9, 123.9

    aspect_ratio = float(IMG_HEIGHT / IMG_WIDTH)  # 384 / 128 = 3.0 or leaf contour aspect ratio
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(h / max(1, w))
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(cnt)
        solidity = float(cnt_area / max(1, hull_area))
    else:
        solidity = 0.77

    lesion_pixel_count = float(np.count_nonzero(lesion_mask))

    part4 = [
        float(leaf_area_ratio),
        float(lesion_area_ratio),
        lesion_b,
        lesion_g,
        lesion_r,
        lesion_h,
        lesion_s,
        lesion_v,
        lesion_l,
        aspect_ratio,
        solidity,
        lesion_pixel_count,
    ]
    features.extend(part4)  # 12
    # Grand Total = 38 + 30 + 60 + 12 = 140 features!

    feature_vec = np.array(features, dtype=np.float32)

    # Calculate summary metrics for API response
    dom_h = float(np.mean(h_pixels))
    dom_s = float(np.mean(s_pixels))
    dom_v = float(np.mean(v_pixels))
    glcm_contrast_mean = float(np.mean(glcm_contrast))
    lesion_density = float(lesion_area_ratio * 1000.0)

    feature_summary = {
        "leaf_area_ratio": round(leaf_area_ratio, 4),
        "lesion_area_ratio": round(lesion_area_ratio, 4),
        "lesion_density_per_1k_px": round(lesion_density, 2),
        "estimated_lesion_blobs": int(lesion_blob_count),
        "dominant_hsv_mean": [round(dom_h, 1), round(dom_s, 1), round(dom_v, 1)],
        "glcm_contrast_mean": round(glcm_contrast_mean, 3),
    }

    return feature_vec, feature_summary
