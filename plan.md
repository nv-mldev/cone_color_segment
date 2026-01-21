# Technical Specification: Pattern Recognition for Cone-Tip Inspection

**Status:** Approved | **Project:** PixIQ 3D / Industrial Vision Controller

---

## 1. Executive Summary

The objective is to identify varied color patterns (Solid, Half-and-Half, or Slanted Strips) on a circular cone tip manufactured from noisy, printed cardboard. Due to high texture noise and overlapping spectral signatures (e.g., Brown vs. Violet), traditional HSV thresholding and unsupervised clustering (K-Means) are avoided.

We utilize a **Supervised Statistical Fingerprinting** approach leveraging **CIELAB Color Space** and **Polar Geometry**.

---

## 2. Core Technical Principles

### A. Color Space: CIELAB ($a^*b^*$)

* **Intensity Independence:** The $L^*$ channel (Lightness) is discarded to eliminate errors caused by reflections and varying ambient light.
* **Spectral signature:** Focus is on $a^*$ (Green-Red) and $b^*$ (Blue-Yellow).

### B. Geometry: Polar Transformation

* **Warping:** The circular tip is unrolled into a rectangular strip using `cv2.warpPolar`.
* **Invariance:** Makes the system invariant to rotation.
* **80% Crop Strategy:** Discards the central 10% (pixel stretching) and outer 10% (perspective distortion).

### C. The "Golden Sequence" of Operations

To maintain data integrity, the order of operations is:

1. **Bilateral Filter (on BGR):** Removes cardboard grain while preserving slanted edges.
2. **CIELAB Conversion:** Mapping clean RGB data to $a^*b^*$.
3. **Polar Warp:** Geometric unrolling of the "cleaned" color data.
4. **Signature Extraction:** Statistical aggregation of the "sweet spot."

---

## 3. Detailed Algorithm Workflow

### Phase 1: Training (Template Generation)

1. **Preprocessing:** Apply the Golden Sequence to 10–20 samples.
2. **Signature Extraction:**
    * **2D Joint Histogram:** Generate a $32 \times 32$ bin histogram of $a$ and $b$.
    * **Normalization:** Scale the histogram so the total sum is $1.0$.
    * **Entropy Calculation:** Measure complexity to distinguish solid colors from patterns.
3. **Storage:** Save the Histogram and Entropy as a JSON/NumPy entry indexed by **PLC Pattern ID**.

### Phase 2: Inspection (Live Runtime)

1. **Trigger:** Receive **Pattern ID** from PLC and pull the Master Template.
2. **Live Feature Extraction:** Apply the Golden Sequence and calculate current Histogram and Entropy.
3. **Statistical Comparison:**
    * **Color Match:** Calculate **Bhattacharyya Distance** between Histograms.
    * **Pattern Match:** Compare absolute difference in Entropy.
4. **Decision Logic:**
    * `IF (Bhattacharyya_Dist < Threshold) AND (Entropy_Delta < Tolerance): PASS`
    * `ELSE: FAIL`

---

## 4. Engineering Comparison

| Challenge | Solution | Engineering Benefit |
| :--- | :--- | :--- |
| **Noisy Cardboard** | Aggregation | Cancels out random texture noise. |
| **Slanted Strips** | Histogramming | Angle invariant; counts color volume regardless of slant. |
| **Overlapping Colors** | Supervised Distance | Avoids "cluster merging" issues. |
| **Rotation** | Polar Warp | Consistent signature regardless of orientation. |

---

## 5. Reference Implementation (Python)

```python
import cv2
import numpy as np

def get_statistical_signature(lab_patch):
    """Generates the 2D Histogram and Entropy signature."""
    hist = cv2.calcHist([lab_patch], [1, 2], None, [32, 32], [0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    marginal_h = cv2.calcHist([lab_patch], [1], None, [256], [0, 256])
    marginal_h /= (marginal_h.sum() + 1e-7)
    entropy = -np.sum(marginal_h * np.log2(marginal_h + 1e-7))
    
    return hist, entropy

def match_pattern(live_hist, live_ent, master_hist, master_ent):
    """Bhattacharyya Distance: 0 (perfect) to 1 (no match)"""
    dist = cv2.compareHist(live_hist, master_hist, cv2.HISTCMP_BHATTACHARYYA)
    return (dist < 0.15) and (abs(live_ent - master_ent) < 0.5)
