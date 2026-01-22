import json
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# ANALYZE FAILED IMAGES
# ============================================================

RESULTS_FILE = "test_results.json"
TEST_BASE_DIR = "test"

if __name__ == "__main__":
    # Load test results
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    # Filter failed images
    failed_images = [r for r in results['detailed_results'] if r['status'] == 'FAIL']

    print("=" * 80)
    print("FAILED IMAGES ANALYSIS")
    print("=" * 80)
    print(f"\nTotal failed: {len(failed_images)} out of {results['total_images']} images\n")

    # Group by class
    failures_by_class = {}
    for img in failed_images:
        class_id = img['class']
        if class_id not in failures_by_class:
            failures_by_class[class_id] = []
        failures_by_class[class_id].append(img)

    # Print detailed list
    for class_id in sorted(failures_by_class.keys()):
        failures = failures_by_class[class_id]
        print(f"\nClass {class_id}: {len(failures)} failures")
        print("-" * 80)
        print(f"{'Image':<20} {'Bhatt Dist':<15} {'Entropy Δ':<15} {'Confidence':<12} {'Illum Warn'}")
        print("-" * 80)

        for f in failures:
            illum = "YES" if f['illumination_warning'] else "NO"
            print(f"{f['image']:<20} {f['bhatt_distance']:<15.4f} {f['entropy_delta']:<15.4f} {f['confidence']:<12}% {illum}")

        # Print file paths for easy inspection
        print(f"\nFile paths for class {class_id} failures:")
        for f in failures:
            img_path = os.path.join(TEST_BASE_DIR, class_id, f['image'])
            print(f"  {img_path}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\n1. Inspect these images visually - they may be:")
    print("   - Wrong class labels (data leakage)")
    print("   - Single color/blank images")
    print("   - Corrupted images")
    print("   - Genuine outliers/defects")
    print("\n2. Check if the same pattern exists in the training folder")
    print("\n3. Consider removing mislabeled images from both train/ and test/")
    print("\n4. Re-train templates after cleaning the data")
