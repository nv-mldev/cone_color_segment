import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.extract_signature import extract_signature_from_image
from utils.bhattacharyya_distance import compute_bhattacharyya_distance

# ============================================================
# ANALYZE TRAINING DATA FOR INCONSISTENT PATTERNS
# ============================================================

TRAIN_BASE_DIR = "train"


def analyze_class(class_id):
    """Analyze internal consistency of a class's training data."""
    class_dir = os.path.join(TRAIN_BASE_DIR, class_id)
    train_images = sorted(glob.glob(os.path.join(class_dir, "*.png")))

    print(f"\n{'=' * 80}")
    print(f"Class {class_id}: {len(train_images)} training images")
    print(f"{'=' * 80}")

    # Extract signatures for all images
    signatures = []
    image_names = []

    for img_path in train_images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        sig = extract_signature_from_image(img)
        if sig is None:
            continue

        signatures.append(sig)
        image_names.append(os.path.basename(img_path))

    n = len(signatures)
    print(f"Successfully extracted {n} signatures\n")

    # Compute pairwise distances
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dist = compute_bhattacharyya_distance(
                    signatures[i]['histogram'],
                    signatures[j]['histogram']
                )
                distance_matrix[i, j] = dist

    # Compute average distance from each image to all others
    avg_distances = []
    for i in range(n):
        avg_dist = np.mean([distance_matrix[i, j] for j in range(n) if i != j])
        avg_distances.append({
            'image': image_names[i],
            'avg_distance': avg_dist,
            'max_distance': np.max(distance_matrix[i, :]),
            'entropy': signatures[i]['entropy'],
            'mean_L': signatures[i]['mean_L']
        })

    # Sort by average distance (outliers have high avg distance)
    avg_distances.sort(key=lambda x: x['avg_distance'], reverse=True)

    # Print results
    print(f"{'Image':<25} {'Avg Dist':<12} {'Max Dist':<12} {'Entropy':<12} {'Mean L*'}")
    print("-" * 80)

    for item in avg_distances:
        marker = " ⚠ OUTLIER" if item['avg_distance'] > 0.3 else ""
        print(f"{item['image']:<25} {item['avg_distance']:<12.4f} {item['max_distance']:<12.4f} {item['entropy']:<12.4f} {item['mean_L']:<12.1f}{marker}")

    # Identify potential outliers
    outliers = [item for item in avg_distances if item['avg_distance'] > 0.3]

    if outliers:
        print(f"\n⚠ POTENTIAL OUTLIERS ({len(outliers)} images):")
        for item in outliers:
            img_path = os.path.join(class_dir, item['image'])
            print(f"  - {img_path}")
            print(f"    Avg distance to other class {class_id} images: {item['avg_distance']:.4f}")

    # Check for bimodal distribution (two distinct groups)
    print(f"\nDistance statistics:")
    print(f"  Mean: {np.mean(distance_matrix[distance_matrix > 0]):.4f}")
    print(f"  Std:  {np.std(distance_matrix[distance_matrix > 0]):.4f}")
    print(f"  Min:  {np.min(distance_matrix[distance_matrix > 0]):.4f}")
    print(f"  Max:  {np.max(distance_matrix):.4f}")

    return {
        'class_id': class_id,
        'image_names': image_names,
        'distance_matrix': distance_matrix,
        'outliers': outliers
    }


if __name__ == "__main__":
    print("=" * 80)
    print("ANALYZING TRAINING DATA CONSISTENCY")
    print("=" * 80)
    print("\nThis checks if training images within each class are similar to each other.")
    print("High average distances indicate potential outliers or mixed patterns.\n")

    # Get all classes
    class_dirs = sorted([d for d in os.listdir(TRAIN_BASE_DIR)
                        if os.path.isdir(os.path.join(TRAIN_BASE_DIR, d))])

    all_results = {}

    for class_id in class_dirs:
        result = analyze_class(class_id)
        all_results[class_id] = result

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_outliers = sum(len(r['outliers']) for r in all_results.values())

    if total_outliers > 0:
        print(f"\nFound {total_outliers} potential outliers across all classes:")
        for class_id, result in all_results.items():
            if result['outliers']:
                print(f"\n  Class {class_id}: {len(result['outliers'])} outliers")
                for outlier in result['outliers']:
                    print(f"    - {outlier['image']}")
    else:
        print("\nNo obvious outliers found. Training data appears consistent.")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\n1. Review images marked as ⚠ OUTLIER - they may be:")
    print("   - Wrong class labels")
    print("   - Single color/blank images")
    print("   - Different pattern variants")
    print("\n2. If outliers are confirmed as wrong, remove them from train/ folder")
    print("\n3. Re-run train_all_templates.py to create cleaner templates")
    print("\n4. Re-run test_all_images.py to check improved accuracy")
