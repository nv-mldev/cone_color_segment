import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.load_template import load_template
from utils.extract_signature import extract_signature_from_image
from utils.bhattacharyya_distance import compute_bhattacharyya_distance

# ============================================================
# CONFIGURATION
# ============================================================

TEST_BASE_DIR = "test"
TEMPLATE_DIR = "templates"

# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def compute_metrics(test_dir, template, class_id):
    """Compute Bhattacharyya distances and entropy deltas for all images in a class."""
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    params = template['preprocess_params']

    metrics = {
        'bhatt_distances': [],
        'entropy_deltas': [],
        'image_names': [],
        'class_id': class_id
    }

    for img_path in test_images:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            continue

        sig = extract_signature_from_image(
            img,
            inner_crop_pct=params['inner_crop_pct'],
            outer_crop_pct=params['outer_crop_pct'],
            bilateral_d=params['bilateral_d'],
            bilateral_sigma_color=params['bilateral_sigma_color'],
            bilateral_sigma_space=params['bilateral_sigma_space']
        )

        if sig is None:
            continue

        bhatt_dist = compute_bhattacharyya_distance(sig['histogram'], template['histogram'])
        entropy_delta = abs(sig['entropy'] - template['entropy'])

        metrics['bhatt_distances'].append(bhatt_dist)
        metrics['entropy_deltas'].append(entropy_delta)
        metrics['image_names'].append(img_name)

    return metrics


if __name__ == "__main__":
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(TEST_BASE_DIR)
                        if os.path.isdir(os.path.join(TEST_BASE_DIR, d))])

    print(f"Analyzing {len(class_dirs)} classes for threshold optimization...")
    print("=" * 80)

    # Store metrics for each class
    all_intra_class_bhatt = []  # Same class (should be low)
    all_intra_class_entropy = []  # Same class (should be low)

    all_inter_class_bhatt = []  # Different class (should be high)
    all_inter_class_entropy = []  # Different class (should be high)

    class_metrics = {}
    templates = {}

    # Load all templates and compute metrics
    for class_id in class_dirs:
        test_dir = os.path.join(TEST_BASE_DIR, class_id)
        template_path = os.path.join(TEMPLATE_DIR, f"class_{class_id}")

        if not os.path.exists(f"{template_path}_meta.json"):
            print(f"Skipping class {class_id} - template not found")
            continue

        print(f"Processing class {class_id}...")
        template = load_template(template_path)
        templates[class_id] = template

        metrics = compute_metrics(test_dir, template, class_id)
        class_metrics[class_id] = metrics

        # Intra-class: test images of same class vs their template
        all_intra_class_bhatt.extend(metrics['bhatt_distances'])
        all_intra_class_entropy.extend(metrics['entropy_deltas'])

        print(f"  Class {class_id}: {len(metrics['bhatt_distances'])} test images")
        print(f"    Bhatt distance: mean={np.mean(metrics['bhatt_distances']):.4f}, std={np.std(metrics['bhatt_distances']):.4f}")
        print(f"    Entropy delta: mean={np.mean(metrics['entropy_deltas']):.4f}, std={np.std(metrics['entropy_deltas']):.4f}")

    # Compute inter-class metrics (test images vs wrong templates)
    print("\nComputing inter-class metrics (cross-validation)...")
    for test_class_id in class_metrics.keys():
        test_dir = os.path.join(TEST_BASE_DIR, test_class_id)
        test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))

        for template_class_id in templates.keys():
            if test_class_id == template_class_id:
                continue  # Skip same class

            template = templates[template_class_id]
            params = template['preprocess_params']

            for img_path in test_images:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                sig = extract_signature_from_image(
                    img,
                    inner_crop_pct=params['inner_crop_pct'],
                    outer_crop_pct=params['outer_crop_pct'],
                    bilateral_d=params['bilateral_d'],
                    bilateral_sigma_color=params['bilateral_sigma_color'],
                    bilateral_sigma_space=params['bilateral_sigma_space']
                )

                if sig is None:
                    continue

                bhatt_dist = compute_bhattacharyya_distance(sig['histogram'], template['histogram'])
                entropy_delta = abs(sig['entropy'] - template['entropy'])

                all_inter_class_bhatt.append(bhatt_dist)
                all_inter_class_entropy.append(entropy_delta)

    # Convert to numpy arrays
    intra_bhatt = np.array(all_intra_class_bhatt)
    intra_entropy = np.array(all_intra_class_entropy)
    inter_bhatt = np.array(all_inter_class_bhatt)
    inter_entropy = np.array(all_inter_class_entropy)

    # Print statistics
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 80)

    print("\nIntra-class (Same Class - Should PASS):")
    print(f"  Bhattacharyya Distance:")
    print(f"    Mean: {np.mean(intra_bhatt):.4f}")
    print(f"    Std:  {np.std(intra_bhatt):.4f}")
    print(f"    Min:  {np.min(intra_bhatt):.4f}")
    print(f"    Max:  {np.max(intra_bhatt):.4f}")
    print(f"    95th percentile: {np.percentile(intra_bhatt, 95):.4f}")

    print(f"\n  Entropy Delta:")
    print(f"    Mean: {np.mean(intra_entropy):.4f}")
    print(f"    Std:  {np.std(intra_entropy):.4f}")
    print(f"    Min:  {np.min(intra_entropy):.4f}")
    print(f"    Max:  {np.max(intra_entropy):.4f}")
    print(f"    95th percentile: {np.percentile(intra_entropy, 95):.4f}")

    print("\nInter-class (Different Class - Should FAIL):")
    print(f"  Bhattacharyya Distance:")
    print(f"    Mean: {np.mean(inter_bhatt):.4f}")
    print(f"    Std:  {np.std(inter_bhatt):.4f}")
    print(f"    Min:  {np.min(inter_bhatt):.4f}")
    print(f"    Max:  {np.max(inter_bhatt):.4f}")
    print(f"    5th percentile: {np.percentile(inter_bhatt, 5):.4f}")

    print(f"\n  Entropy Delta:")
    print(f"    Mean: {np.mean(inter_entropy):.4f}")
    print(f"    Std:  {np.std(inter_entropy):.4f}")
    print(f"    Min:  {np.min(inter_entropy):.4f}")
    print(f"    Max:  {np.max(inter_entropy):.4f}")
    print(f"    5th percentile: {np.percentile(inter_entropy, 5):.4f}")

    # Suggest optimal thresholds
    print("\n" + "=" * 80)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 80)

    # Method 1: Use mean + 2*std of intra-class
    bhatt_thresh_method1 = np.mean(intra_bhatt) + 2 * np.std(intra_bhatt)
    entropy_thresh_method1 = np.mean(intra_entropy) + 2 * np.std(intra_entropy)

    # Method 2: Use 95th percentile of intra-class
    bhatt_thresh_method2 = np.percentile(intra_bhatt, 95)
    entropy_thresh_method2 = np.percentile(intra_entropy, 95)

    # Method 3: Midpoint between intra max and inter min
    bhatt_thresh_method3 = (np.max(intra_bhatt) + np.min(inter_bhatt)) / 2
    entropy_thresh_method3 = (np.max(intra_entropy) + np.min(inter_entropy)) / 2

    print("\nMethod 1: Mean + 2*Std (covers ~95% of intra-class)")
    print(f"  Bhattacharyya threshold: {bhatt_thresh_method1:.4f}")
    print(f"  Entropy threshold: {entropy_thresh_method1:.4f}")

    print("\nMethod 2: 95th Percentile (covers 95% of intra-class)")
    print(f"  Bhattacharyya threshold: {bhatt_thresh_method2:.4f}")
    print(f"  Entropy threshold: {entropy_thresh_method2:.4f}")

    print("\nMethod 3: Midpoint (balance between intra and inter)")
    print(f"  Bhattacharyya threshold: {bhatt_thresh_method3:.4f}")
    print(f"  Entropy threshold: {entropy_thresh_method3:.4f}")

    # Check separation
    overlap_bhatt = (np.min(inter_bhatt) - np.max(intra_bhatt))
    overlap_entropy = (np.min(inter_entropy) - np.max(intra_entropy))

    print("\n" + "=" * 80)
    print("SEPARABILITY ANALYSIS")
    print("=" * 80)
    print(f"Bhattacharyya gap (inter_min - intra_max): {overlap_bhatt:.4f}")
    if overlap_bhatt > 0:
        print("  ✓ Perfect separation! No overlap.")
    else:
        print("  ⚠ Some overlap exists. Thresholds may cause errors.")

    print(f"\nEntropy gap (inter_min - intra_max): {overlap_entropy:.4f}")
    if overlap_entropy > 0:
        print("  ✓ Perfect separation! No overlap.")
    else:
        print("  ⚠ Some overlap exists. Thresholds may cause errors.")

    # Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bhattacharyya distance histogram
    axes[0, 0].hist(intra_bhatt, bins=30, alpha=0.7, label='Intra-class (Same)', color='green')
    axes[0, 0].hist(inter_bhatt, bins=30, alpha=0.7, label='Inter-class (Different)', color='red')
    axes[0, 0].axvline(bhatt_thresh_method2, color='blue', linestyle='--', linewidth=2, label='Threshold (95th %ile)')
    axes[0, 0].set_xlabel('Bhattacharyya Distance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Bhattacharyya Distance Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Entropy delta histogram
    axes[0, 1].hist(intra_entropy, bins=30, alpha=0.7, label='Intra-class (Same)', color='green')
    axes[0, 1].hist(inter_entropy, bins=30, alpha=0.7, label='Inter-class (Different)', color='red')
    axes[0, 1].axvline(entropy_thresh_method2, color='blue', linestyle='--', linewidth=2, label='Threshold (95th %ile)')
    axes[0, 1].set_xlabel('Entropy Delta')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Entropy Delta Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bhattacharyya box plot
    axes[1, 0].boxplot([intra_bhatt, inter_bhatt], labels=['Intra-class', 'Inter-class'])
    axes[1, 0].axhline(bhatt_thresh_method2, color='blue', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 0].set_ylabel('Bhattacharyya Distance')
    axes[1, 0].set_title('Bhattacharyya Distance Box Plot')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Entropy box plot
    axes[1, 1].boxplot([intra_entropy, inter_entropy], labels=['Intra-class', 'Inter-class'])
    axes[1, 1].axhline(entropy_thresh_method2, color='blue', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 1].set_ylabel('Entropy Delta')
    axes[1, 1].set_title('Entropy Delta Box Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=150)
    print("Visualization saved to: threshold_analysis.png")

    plt.show()

    print("\nDone!")
