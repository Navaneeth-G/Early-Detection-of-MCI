"""
MRI Brain Image Validator — Ultra Strict Version
Rejects ANY image that is not a genuine brain MRI scan.
"""

import numpy as np
from PIL import Image


def validate_brain_mri(pil_image: Image.Image):
    """
    Returns (is_valid: bool, reason: str)

    An image passes ONLY if it satisfies ALL of the following:
      1. Nearly grayscale  (no colour)
      2. Large dark background  (MRI scanner black border)
      3. Most pixels are dark   (MRI histogram shape)
      4. Very low colour saturation
      5. Has enough contrast    (not blank)
      6. Has a brighter central region  (brain tissue)
    """

    img_rgb  = pil_image.convert("RGB")
    img_gray = pil_image.convert("L")

    arr      = np.array(img_rgb,  dtype=np.float32)
    gray     = np.array(img_gray, dtype=np.float32)

    R, G, B  = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    failures = []

    # ── 1. Grayscale check ──────────────────────────────────
    # MRI:   R ≈ G ≈ B  →  mean channel diff < 10
    # Photo: colourful   →  mean channel diff >> 10
    color_diff = (np.mean(np.abs(R-G)) +
                  np.mean(np.abs(R-B)) +
                  np.mean(np.abs(G-B))) / 3.0
    if color_diff > 10:
        failures.append(
            f"image is coloured (channel deviation = {color_diff:.1f}). "
            f"Brain MRI scans are always grayscale."
        )

    # ── 2. HSV Saturation check ─────────────────────────────
    # MRI:   saturation ≈ 0
    # Photo: saturation > 0.10 (flowers, faces, nature etc.)
    r_n = R / 255.0
    g_n = G / 255.0
    b_n = B / 255.0
    cmax = np.maximum(np.maximum(r_n, g_n), b_n)
    cmin = np.minimum(np.minimum(r_n, g_n), b_n)
    sat  = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-6), 0)
    mean_sat = float(np.mean(sat))
    if mean_sat > 0.10:
        failures.append(
            f"image has colour saturation = {mean_sat:.3f}. "
            f"Brain MRI scans must have saturation < 0.10."
        )

    # ── 3. Dark background ratio ────────────────────────────
    # MRI:   >25% of pixels are near-black  (scanner background)
    # Photo: usually <10% near-black pixels
    dark_ratio = float(np.sum(gray < 15) / gray.size)
    if dark_ratio < 0.20:
        failures.append(
            f"only {dark_ratio*100:.1f}% dark pixels found. "
            f"MRI scans need >20% black scanner background."
        )

    # ── 4. Pixel intensity histogram shape ─────────────────
    # MRI:   bimodal — large dark peak (bg) + smaller bright peak (tissue)
    #        → lowest 2 bins (0-51) must hold > 30% of all pixels
    # Photo: roughly uniform or bright-skewed
    hist, _ = np.histogram(gray, bins=10, range=(0,255))
    dark_frac = float((hist[0] + hist[1]) / gray.size)
    if dark_frac < 0.25:
        failures.append(
            f"pixel histogram not MRI-like ({dark_frac*100:.1f}% dark pixels in "
            f"lowest bins). MRI needs >25%."
        )

    # ── 5. Contrast / not blank ─────────────────────────────
    std_val = float(np.std(gray))
    if std_val < 20:
        failures.append(
            f"image appears blank or has too little contrast "
            f"(std = {std_val:.1f}; need > 20)."
        )

    # ── 6. Bright central region ────────────────────────────
    # The brain tissue sits in the centre and is brighter than the border.
    h, w   = gray.shape
    cy, cx = h // 2, w // 2
    ry, rx = h // 4, w // 4
    yy, xx = np.ogrid[:h, :w]
    mask   = ((xx - cx)**2 / max(rx,1)**2 + (yy - cy)**2 / max(ry,1)**2) <= 1.0
    c_mean = float(np.mean(gray[mask]))
    b_mean = float(np.mean(gray[~mask]))
    if c_mean <= b_mean + 5:
        failures.append(
            f"no bright central brain region detected "
            f"(center mean={c_mean:.1f}, border mean={b_mean:.1f}). "
            f"MRI scans have brighter tissue in the centre."
        )

    # ── Decision ────────────────────────────────────────────
    if not failures:
        return True, "Valid brain MRI scan."

    bullet_list = "\n".join(f"• {f}" for f in failures)
    reason = (
        "This image does not appear to be a brain MRI scan.\n\n"
        + bullet_list
    )
    return False, reason
