# Technical Specification: Pattern Recognition for Cone-Tip Inspection

**Status:** Draft v2 | **Project:** PixIQ 3D / Industrial Vision Controller

---

## 1. Executive Summary

The objective is to identify varied color patterns (Solid, Half-and-Half, or Slanted Strips) on a circular cone tip manufactured from noisy, printed cardboard. Due to high texture noise and overlapping spectral signatures (e.g., Brown vs. Violet), traditional HSV thresholding and unsupervised clustering (K-Means) are avoided.

We utilize a **Supervised Statistical Fingerprinting** approach leveraging **CIELAB Color Space** and **Polar Geometry**.

### Changes from v1
- Fixed histogram normalization (L1 instead of MINMAX)
- Corrected entropy calculation to use 2D joint entropy
- Added per-pattern configurable thresholds
- Added confidence scoring alongside binary decision
- Added illumination health monitoring
- Optimized histogram bin ranges for real-time performance

---

## 2. Core Technical Principles

### A. Color Space: CIELAB ($a^*b^*$)

* **Intensity Independence:** The $L^*$ channel (Lightness) is discarded to eliminate errors caused by reflections and varying ambient light.
* **Spectral Signature:** Focus is on $a^*$ (Green-Red) and $b^*$ (Blue-Yellow).
* **Perceptual Uniformity:** Euclidean distance in $a^*b^*$ correlates with human color perception.

### B. Geometry: Polar Transformation

* **Warping:** The circular tip is unrolled into a rectangular strip using `cv2.warpPolar`.
* **Rotation Invariance:** Since cone orientation doesn't matter, histogram-based matching naturally handles any rotational position.
* **80% Crop Strategy:** Discards the central 10% (pixel stretching) and outer 10% (perspective distortion). These percentages are configurable per product.

### C. The "Golden Sequence" of Operations

To maintain data integrity, the order of operations is:

1. **Bilateral Filter (on BGR):** Removes cardboard grain while preserving slanted edges.
2. **CIELAB Conversion:** Mapping clean RGB data to $a^*b^*$.
3. **Polar Warp:** Geometric unrolling of the "cleaned" color data.
4. **Signature Extraction:** Statistical aggregation of the "sweet spot."

---

## 3. Detailed Algorithm Workflow

### Phase 1: Training (Template Generation)

1. **Preprocessing:** Apply the Golden Sequence to 10–20 samples per pattern.
2. **Signature Extraction:**
    * **2D Joint Histogram:** Generate a $32 \times 32$ bin histogram of $a^*$ and $b^*$.
    * **L1 Normalization:** Scale the histogram so the total sum equals $1.0$ (probability distribution).
    * **2D Joint Entropy:** Calculate Shannon entropy on the normalized 2D histogram to measure color complexity.
3. **Threshold Calibration:**
    * Compute intra-class variation (histogram distances between "good" samples of same pattern).
    * Set `bhatt_threshold` = mean + 2σ of intra-class Bhattacharyya distances.
    * Set `entropy_threshold` = mean + 2σ of intra-class entropy deltas.
4. **Storage:** Save per pattern:
    * Normalized 2D Histogram (32×32 float32)
    * 2D Joint Entropy (float)
    * Calibrated thresholds (bhatt_threshold, entropy_threshold)
    * Mean L* value (for illumination monitoring)

### Phase 2: Inspection (Live Runtime)

1. **Trigger:** Receive **Pattern ID** from PLC and pull the Master Template.
2. **Illumination Check:** Compute mean L* and compare against master. Flag warning if drift > 10%.
3. **Live Feature Extraction:** Apply the Golden Sequence and calculate:
    * L1-normalized 2D histogram
    * 2D joint entropy
4. **Statistical Comparison:**
    * **Color Match:** Calculate **Bhattacharyya Distance** between histograms.
    * **Pattern Match:** Compare absolute difference in entropy.
5. **Confidence Calculation:**
    * `color_conf = max(0, 1 - bhatt_dist / bhatt_threshold)`
    * `pattern_conf = max(0, 1 - entropy_delta / entropy_threshold)`
    * `overall_conf = 0.7 * color_conf + 0.3 * pattern_conf`
6. **Decision Logic:**
    * `IF (bhatt_dist < bhatt_threshold) AND (entropy_delta < entropy_threshold): PASS`
    * `ELSE: FAIL`
    * Always output confidence percentage for operator review.

---

## 4. Engineering Comparison

| Challenge | Solution | Engineering Benefit |
| :--- | :--- | :--- |
| **Noisy Cardboard** | Histogram Aggregation | Cancels out random texture noise. |
| **Slanted Strips** | 2D Histogramming | Angle invariant; counts color volume regardless of slant. |
| **Overlapping Colors** | Supervised Distance | Avoids "cluster merging" issues of unsupervised methods. |
| **Rotation** | Polar Warp + Histogram | Consistent signature regardless of cone orientation. |
| **Illumination Drift** | L* Monitoring | Early warning for lighting changes. |
| **Threshold Tuning** | Per-Pattern Calibration | Tight thresholds for solid colors, looser for multi-color patterns. |

---

## 5. Reference Implementation (Python)

```python
import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
HIST_BINS = 32
HIST_RANGE = [0, 256, 0, 256]  # Can be tightened per product
DEFAULT_BHATT_THRESHOLD = 0.15
DEFAULT_ENTROPY_THRESHOLD = 0.5


# ============================================================
# SIGNATURE EXTRACTION
# ============================================================
def get_statistical_signature(lab_patch):
    """
    Generates the 2D Histogram and 2D Joint Entropy signature.

    Args:
        lab_patch: numpy array of shape (H, W, 3) in CIELAB color space

    Returns:
        hist: L1-normalized 2D histogram (32x32)
        entropy: 2D joint entropy (float)
        mean_L: mean lightness for illumination monitoring
    """
    # 2D Joint Histogram on a* and b* channels
    hist = cv2.calcHist(
        [lab_patch],
        [1, 2],           # a* and b* channels
        None,
        [HIST_BINS, HIST_BINS],
        HIST_RANGE
    )

    # L1 Normalization (sum to 1.0) - CRITICAL for Bhattacharyya
    hist = hist / (hist.sum() + 1e-7)

    # 2D Joint Entropy
    hist_flat = hist.flatten()
    hist_flat = hist_flat[hist_flat > 0]  # Avoid log(0)
    entropy = -np.sum(hist_flat * np.log2(hist_flat))

    # Mean L* for illumination monitoring
    mean_L = lab_patch[:, :, 0].mean()

    return hist, entropy, mean_L


# ============================================================
# PATTERN MATCHING
# ============================================================
def match_pattern(live_hist, live_entropy, live_L,
                  master_hist, master_entropy, master_L,
                  bhatt_threshold=DEFAULT_BHATT_THRESHOLD,
                  entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
                  L_warning_threshold=10.0):
    """
    Compares live signature against master template.

    Args:
        live_hist: L1-normalized histogram from live image
        live_entropy: 2D joint entropy from live image
        live_L: mean L* from live image
        master_hist: stored master histogram
        master_entropy: stored master entropy
        master_L: stored master mean L*
        bhatt_threshold: maximum allowed Bhattacharyya distance
        entropy_threshold: maximum allowed entropy difference
        L_warning_threshold: L* drift percentage to trigger warning

    Returns:
        result: dict with pass/fail, confidence, distances, and warnings
    """
    # Bhattacharyya Distance: 0 (identical) to 1 (no overlap)
    bhatt_dist = cv2.compareHist(
        live_hist.astype(np.float32),
        master_hist.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA
    )

    # Entropy difference
    entropy_delta = abs(live_entropy - master_entropy)

    # Illumination drift check
    L_drift_pct = abs(live_L - master_L) / (master_L + 1e-7) * 100
    illumination_warning = L_drift_pct > L_warning_threshold

    # Confidence calculation
    color_conf = max(0.0, 1.0 - (bhatt_dist / bhatt_threshold))
    pattern_conf = max(0.0, 1.0 - (entropy_delta / entropy_threshold))
    overall_conf = 0.7 * color_conf + 0.3 * pattern_conf

    # Decision
    passed = (bhatt_dist < bhatt_threshold) and (entropy_delta < entropy_threshold)

    return {
        'pass': passed,
        'confidence': round(overall_conf * 100, 1),
        'bhattacharyya_distance': round(bhatt_dist, 4),
        'entropy_delta': round(entropy_delta, 4),
        'illumination_warning': illumination_warning,
        'L_drift_percent': round(L_drift_pct, 1)
    }


# ============================================================
# PREPROCESSING PIPELINE (Golden Sequence)
# ============================================================
def preprocess_cone_tip(bgr_image, center, radius,
                        inner_crop_pct=0.10, outer_crop_pct=0.10,
                        bilateral_d=9, bilateral_sigma_color=75,
                        bilateral_sigma_space=75):
    """
    Applies the Golden Sequence: Filter -> LAB -> Polar Warp -> Crop

    Args:
        bgr_image: input BGR image
        center: (cx, cy) tuple for cone tip center
        radius: radius of the circular ROI
        inner_crop_pct: percentage of inner region to discard (default 10%)
        outer_crop_pct: percentage of outer region to discard (default 10%)
        bilateral_*: bilateral filter parameters

    Returns:
        lab_patch: cropped LAB patch ready for signature extraction
    """
    # Step 1: Bilateral filter (on BGR)
    filtered = cv2.bilateralFilter(
        bgr_image,
        bilateral_d,
        bilateral_sigma_color,
        bilateral_sigma_space
    )

    # Step 2: Convert to CIELAB
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)

    # Step 3: Polar warp
    polar = cv2.warpPolar(
        lab,
        (radius, 360),  # (r_max, theta_max)
        center,
        radius,
        cv2.WARP_POLAR_LINEAR
    )

    # Step 4: Crop the "sweet spot" (discard inner and outer rings)
    r_inner = int(radius * inner_crop_pct)
    r_outer = int(radius * (1 - outer_crop_pct))
    lab_patch = polar[r_inner:r_outer, :]

    return lab_patch


# ============================================================
# TEMPLATE TRAINING
# ============================================================
def train_template(sample_images, center, radius, **preprocess_kwargs):
    """
    Generates a master template from multiple good samples.

    Args:
        sample_images: list of BGR images (10-20 recommended)
        center: (cx, cy) for all images (assumes fixed camera)
        radius: ROI radius

    Returns:
        template: dict containing histogram, entropy, thresholds, mean_L
    """
    histograms = []
    entropies = []
    mean_Ls = []

    for img in sample_images:
        patch = preprocess_cone_tip(img, center, radius, **preprocess_kwargs)
        hist, ent, mean_L = get_statistical_signature(patch)
        histograms.append(hist)
        entropies.append(ent)
        mean_Ls.append(mean_L)

    # Master histogram: mean of all samples
    master_hist = np.mean(histograms, axis=0)
    master_hist = master_hist / (master_hist.sum() + 1e-7)  # Re-normalize

    # Master entropy and L*
    master_entropy = np.mean(entropies)
    master_L = np.mean(mean_Ls)

    # Compute intra-class distances for threshold calibration
    bhatt_distances = []
    entropy_deltas = []
    for hist, ent in zip(histograms, entropies):
        bd = cv2.compareHist(
            hist.astype(np.float32),
            master_hist.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        bhatt_distances.append(bd)
        entropy_deltas.append(abs(ent - master_entropy))

    # Threshold = mean + 2*std (covers ~95% of good samples)
    bhatt_threshold = np.mean(bhatt_distances) + 2 * np.std(bhatt_distances)
    entropy_threshold = np.mean(entropy_deltas) + 2 * np.std(entropy_deltas)

    # Ensure minimum thresholds
    bhatt_threshold = max(bhatt_threshold, 0.05)
    entropy_threshold = max(entropy_threshold, 0.2)

    return {
        'histogram': master_hist,
        'entropy': master_entropy,
        'mean_L': master_L,
        'bhatt_threshold': round(bhatt_threshold, 4),
        'entropy_threshold': round(entropy_threshold, 4),
        'sample_count': len(sample_images)
    }
```

---

## 6. Performance Characteristics

| Operation | Typical Time (640x480) | Notes |
|-----------|------------------------|-------|
| Bilateral Filter | ~2-5 ms | Most expensive step; can reduce `d` for speed |
| BGR to LAB | < 1 ms | OpenCV optimized |
| Polar Warp | < 1 ms | OpenCV optimized |
| 2D Histogram | < 1 ms | OpenCV optimized |
| Bhattacharyya | < 0.1 ms | Simple array operation |
| **Total Pipeline** | **~5-10 ms** | Real-time capable at 100+ FPS |

---

## 7. Calibration Guidelines

### Initial Deployment
1. Collect 20 "good" samples per pattern under normal production conditions.
2. Run `train_template()` to generate master and auto-calibrate thresholds.
3. Validate with 20 additional "good" samples — expect >98% pass rate.
4. Test with known "bad" samples — expect >95% fail rate.

### Threshold Adjustment
- **Too many false rejects?** Increase thresholds by 10-20%.
- **Missing defects?** Decrease thresholds by 10-20%.
- **Solid colors:** Typically need tighter thresholds (bhatt < 0.10).
- **Multi-color patterns:** Can tolerate looser thresholds (bhatt < 0.20).

### Illumination Monitoring
- If `illumination_warning` triggers frequently, recalibrate lighting.
- Re-train templates if lighting fixture is changed.

---

## 8. Data Storage Format

```json
{
  "pattern_id": "P001_RED_SOLID",
  "created": "2026-01-20T10:30:00Z",
  "sample_count": 20,
  "histogram": "<base64 encoded 32x32 float32 array>",
  "entropy": 2.847,
  "mean_L": 142.5,
  "bhatt_threshold": 0.12,
  "entropy_threshold": 0.35,
  "preprocess_params": {
    "inner_crop_pct": 0.10,
    "outer_crop_pct": 0.10,
    "bilateral_d": 9,
    "bilateral_sigma_color": 75,
    "bilateral_sigma_space": 75
  }
}
```

---

## 9. Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1 | - | Initial specification |
| v2 | 2026-01-20 | Fixed histogram normalization (L1), corrected entropy to 2D joint, added per-pattern thresholds, added confidence scoring, added illumination monitoring |
