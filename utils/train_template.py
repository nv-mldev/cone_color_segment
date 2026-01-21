import numpy as np
from datetime import datetime

from .extract_signature import extract_signature_from_image
from .bhattacharyya_distance import compute_bhattacharyya_distance


def train_template(sample_images, pattern_id,
                   inner_crop_pct=0.10, outer_crop_pct=0.10,
                   bilateral_d=9, bilateral_sigma_color=75,
                   bilateral_sigma_space=75,
                   min_threshold_bhatt=0.05, min_threshold_entropy=0.2):
    """
    Generate a master template from multiple good samples.

    Args:
        sample_images: List of BGR images (10-20 recommended)
        pattern_id: Unique identifier for this pattern
        inner_crop_pct: Inner crop percentage
        outer_crop_pct: Outer crop percentage
        bilateral_*: Bilateral filter parameters
        min_threshold_bhatt: Minimum Bhattacharyya threshold
        min_threshold_entropy: Minimum entropy threshold

    Returns:
        Template dict containing all training data
    """
    histograms = []
    entropies = []
    mean_Ls = []

    for i, img in enumerate(sample_images):
        sig = extract_signature_from_image(
            img,
            inner_crop_pct=inner_crop_pct,
            outer_crop_pct=outer_crop_pct,
            bilateral_d=bilateral_d,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space
        )

        if sig is None:
            print(f"Warning: Could not extract signature from sample {i}")
            continue

        histograms.append(sig['histogram'])
        entropies.append(sig['entropy'])
        mean_Ls.append(sig['mean_L'])

    if len(histograms) < 2:
        raise ValueError("Need at least 2 valid samples to train a template")

    # Master histogram: mean of all samples
    master_hist = np.mean(histograms, axis=0)
    master_hist = master_hist / (master_hist.sum() + 1e-7)  # Re-normalize

    # Master entropy and L*
    master_entropy = float(np.mean(entropies))
    master_L = float(np.mean(mean_Ls))

    # Compute intra-class distances for threshold calibration
    bhatt_distances = []
    entropy_deltas = []

    for hist, ent in zip(histograms, entropies):
        bd = compute_bhattacharyya_distance(hist, master_hist)
        bhatt_distances.append(bd)
        entropy_deltas.append(abs(ent - master_entropy))

    # Threshold = mean + 2*std (covers ~95% of good samples)
    bhatt_mean = np.mean(bhatt_distances)
    bhatt_std = np.std(bhatt_distances)
    bhatt_threshold = bhatt_mean + 2 * bhatt_std

    entropy_mean = np.mean(entropy_deltas)
    entropy_std = np.std(entropy_deltas)
    entropy_threshold = entropy_mean + 2 * entropy_std

    # Ensure minimum thresholds
    bhatt_threshold = max(float(bhatt_threshold), min_threshold_bhatt)
    entropy_threshold = max(float(entropy_threshold), min_threshold_entropy)

    template = {
        'pattern_id': pattern_id,
        'created': datetime.now().isoformat(),
        'sample_count': len(histograms),
        'histogram': master_hist,
        'entropy': master_entropy,
        'mean_L': master_L,
        'bhatt_threshold': round(bhatt_threshold, 4),
        'entropy_threshold': round(entropy_threshold, 4),
        'preprocess_params': {
            'inner_crop_pct': inner_crop_pct,
            'outer_crop_pct': outer_crop_pct,
            'bilateral_d': bilateral_d,
            'bilateral_sigma_color': bilateral_sigma_color,
            'bilateral_sigma_space': bilateral_sigma_space
        },
        'calibration_stats': {
            'bhatt_mean': round(float(bhatt_mean), 4),
            'bhatt_std': round(float(bhatt_std), 4),
            'entropy_mean': round(float(entropy_mean), 4),
            'entropy_std': round(float(entropy_std), 4)
        }
    }

    return template
