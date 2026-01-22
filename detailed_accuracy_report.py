import os
import cv2
import glob
import numpy as np
from utils.load_template import load_template
from utils.extract_signature import extract_signature_from_image
from utils.bhattacharyya_distance import compute_bhattacharyya_distance

# ============================================================
# DETAILED ACCURACY REPORT - THRESHOLD VS NEAREST NEIGHBOR
# ============================================================

TEST_BASE_DIR = "test"
TEMPLATE_DIR = "templates"

if __name__ == "__main__":
    # Get all classes
    class_dirs = sorted([d for d in os.listdir(TEST_BASE_DIR)
                        if os.path.isdir(os.path.join(TEST_BASE_DIR, d))])

    print("=" * 100)
    print("DETAILED ACCURACY REPORT - THRESHOLD vs NEAREST NEIGHBOR")
    print("=" * 100)

    # Load all templates
    templates = {}
    for class_id in class_dirs:
        template_path = os.path.join(TEMPLATE_DIR, f"class_{class_id}")
        if os.path.exists(f"{template_path}_meta.json"):
            templates[class_id] = load_template(template_path)

    print(f"\nLoaded {len(templates)} templates\n")

    # Store results
    all_results = []

    print(f"{'Class':<6} {'Image':<20} {'Own Dist':<12} {'Best Dist':<12} {'Best Class':<12} {'Threshold':<12} {'Thr Pass':<10} {'NN Pass':<10}")
    print("-" * 100)

    # Test each class
    for true_class in class_dirs:
        test_dir = os.path.join(TEST_BASE_DIR, true_class)
        test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
        template = templates[true_class]

        for img_path in test_images:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)

            if img is None:
                continue

            params = template['preprocess_params']
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

            # Distance to own template
            own_dist = compute_bhattacharyya_distance(
                sig['histogram'],
                template['histogram']
            )

            # Find best matching template
            best_class = None
            best_dist = float('inf')

            for test_class, test_template in templates.items():
                test_params = test_template['preprocess_params']
                test_sig = extract_signature_from_image(
                    img,
                    inner_crop_pct=test_params['inner_crop_pct'],
                    outer_crop_pct=test_params['outer_crop_pct'],
                    bilateral_d=test_params['bilateral_d'],
                    bilateral_sigma_color=test_params['bilateral_sigma_color'],
                    bilateral_sigma_space=test_params['bilateral_sigma_space']
                )

                if test_sig is None:
                    continue

                dist = compute_bhattacharyya_distance(
                    test_sig['histogram'],
                    test_template['histogram']
                )

                if dist < best_dist:
                    best_dist = dist
                    best_class = test_class

            # Check threshold-based pass
            threshold = template['bhatt_threshold']
            threshold_pass = own_dist < threshold

            # Check nearest-neighbor pass
            nn_pass = (best_class == true_class)

            # Store result
            result = {
                'true_class': true_class,
                'image': img_name,
                'own_distance': own_dist,
                'best_distance': best_dist,
                'best_class': best_class,
                'threshold': threshold,
                'threshold_pass': threshold_pass,
                'nn_pass': nn_pass
            }
            all_results.append(result)

            # Print
            thr_status = "✓ PASS" if threshold_pass else "✗ FAIL"
            nn_status = "✓ PASS" if nn_pass else "✗ FAIL"

            marker = ""
            if not threshold_pass and nn_pass:
                marker = " ⚠ OUTLIER (correct class but beyond threshold)"
            elif not nn_pass:
                marker = " ⚠⚠ MISCLASSIFIED"

            print(f"{true_class:<6} {img_name:<20} {own_dist:<12.4f} {best_dist:<12.4f} {best_class:<12} {threshold:<12.4f} {thr_status:<10} {nn_status:<10}{marker}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY BY CLASS")
    print("=" * 100)

    print(f"\n{'Class':<8} {'Total':<8} {'Thr Pass':<12} {'Thr Fail':<12} {'NN Pass':<12} {'NN Fail':<12} {'Thr Acc%':<12} {'NN Acc%':<12} {'Outliers'}")
    print("-" * 100)

    total_thr_pass = 0
    total_thr_fail = 0
    total_nn_pass = 0
    total_nn_fail = 0
    total_outliers = 0

    for class_id in class_dirs:
        class_results = [r for r in all_results if r['true_class'] == class_id]
        total = len(class_results)

        thr_pass = sum(1 for r in class_results if r['threshold_pass'])
        thr_fail = total - thr_pass
        nn_pass = sum(1 for r in class_results if r['nn_pass'])
        nn_fail = total - nn_pass
        outliers = sum(1 for r in class_results if not r['threshold_pass'] and r['nn_pass'])

        thr_acc = (thr_pass / total * 100) if total > 0 else 0
        nn_acc = (nn_pass / total * 100) if total > 0 else 0

        print(f"{class_id:<8} {total:<8} {thr_pass:<12} {thr_fail:<12} {nn_pass:<12} {nn_fail:<12} {thr_acc:<12.1f} {nn_acc:<12.1f} {outliers}")

        total_thr_pass += thr_pass
        total_thr_fail += thr_fail
        total_nn_pass += nn_pass
        total_nn_fail += nn_fail
        total_outliers += outliers

    print("-" * 100)

    total_samples = len(all_results)
    overall_thr_acc = (total_thr_pass / total_samples * 100) if total_samples > 0 else 0
    overall_nn_acc = (total_nn_pass / total_samples * 100) if total_samples > 0 else 0

    print(f"{'TOTAL':<8} {total_samples:<8} {total_thr_pass:<12} {total_thr_fail:<12} {total_nn_pass:<12} {total_nn_fail:<12} {overall_thr_acc:<12.1f} {overall_nn_acc:<12.1f} {total_outliers}")

    # Key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    print(f"\n1. THRESHOLD-BASED ACCURACY: {overall_thr_acc:.1f}%")
    print(f"   - {total_thr_pass} images within threshold of their template")
    print(f"   - {total_thr_fail} images beyond threshold (rejected)")

    print(f"\n2. NEAREST-NEIGHBOR ACCURACY: {overall_nn_acc:.1f}%")
    print(f"   - {total_nn_pass} images closest to correct template")
    print(f"   - {total_nn_fail} images closest to wrong template")

    print(f"\n3. OUTLIERS: {total_outliers} images")
    print(f"   - These are CORRECT class but TOO FAR from template")
    print(f"   - They would fail threshold check but pass nearest-neighbor")

    if total_nn_fail > 0:
        print(f"\n4. MISCLASSIFICATIONS: {total_nn_fail} images")
        print(f"   - These images are closer to WRONG template than their own")
        misclassified = [r for r in all_results if not r['nn_pass']]
        for r in misclassified:
            print(f"     • {r['image']} (true: {r['true_class']}, closest: {r['best_class']})")

    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    if overall_nn_acc == 100.0:
        print("\n✓ EXCELLENT: 100% nearest-neighbor accuracy!")
        print("  → All images are closest to their correct template")
        print("  → No cross-class confusion")

        if total_outliers > 0:
            print(f"\n⚠ BUT: {total_outliers} outliers exist (beyond threshold)")
            print("  → These might be:")
            print("    - Genuine variation within the class (adjust threshold)")
            print("    - Data quality issues (clean dataset)")
            print("    - Edge cases / defects (investigate)")
    else:
        print(f"\n⚠ WARNING: {total_nn_fail} images are misclassified (closest to wrong template)")
        print("  → Investigate these images - likely data leakage or mislabeling")

    print()
