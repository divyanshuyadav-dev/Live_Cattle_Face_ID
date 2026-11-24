import cv2
import numpy as np
from core.config import MIN_FACE_SIZE, BLUR_THRESHOLD


def check_crop_quality(crop_bgr: np.ndarray) -> tuple[bool, str, float]:
    """
    Validate face crop size and sharpness.
    Returns: (is_valid, reason, blur_score)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, "Empty crop", 0.0

    h, w = crop_bgr.shape[:2]
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        return False, f"Face too small ({w}x{h}px < {MIN_FACE_SIZE}px)", 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if blur_score < BLUR_THRESHOLD:
        return False, f"Image too blurry (score {blur_score:.1f} < {BLUR_THRESHOLD})", blur_score

    return True, "OK", blur_score