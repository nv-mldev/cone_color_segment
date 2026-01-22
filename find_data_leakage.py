import os
import cv2
import glob
import numpy as np
from utils.extract_signature import extract_signature_from_image
from utils.bhattacharyya_distance import compute_bhattacharyya_distance

# ============================================================
# FIND DATA LEAKAGE - CHECK FOR WRONG PATTERNS IN TRAINING DATA
# ============================================================

TRAIN_BASE_DIR = "train"
TEST_BASE_DIR = "test"

# Failed images from analysis
FAILED_IMAGES = {
    '10': ['6564_vl.png'],
    '5': ['8766_vl.png'],
    '6': ['7468_vl.png'],
    '9': ['7981_vl.png', '8341_vl.png', '9416_vl.png', '9951_vl.png']
}


def extract_sig(img_path):
    """Extract signature from an image."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    return extract_signature_from_image(img)


if __name__ == "__main__":
    print("=" * 80)
    print("CHECKING FOR DATA LEAKAGE IN TRAINING SETS")
    print("=" * 80)
    print("\nThis will check if failed test images are similar to wrong classes")
    print("in the training data (indicating mislabeling or data leakage).\n")

    for test_class in sorted(FAILED_IMAGES.keys()):
        failed_imgs = FAILED_IMAGES[test_class]

        print(f"\n{'=' * 80}")
        print(f"Class {test_class}: Analyzing {len(failed_imgs)} failed images")
        print(f"{'=' * 80}")

        for failed_img_name in failed_imgs:
            failed_img_path = os.path.join(TEST_BASE_DIR, test_class, failed_img_name)

            print(f"\n{failed_img_name}:")
            print(f"  Path: {failed_img_path}")

            # Extract signature from failed image
            failed_sig = extract_sig(failed_img_path)
            if failed_sig is None:
                print(f"  ERROR: Could not extract signature")
                continue

            # Compare against ALL training classes
            best_matches = []

            for train_class in os.listdir(TRAIN_BASE_DIR):
                train_dir = os.path.join(TRAIN_BASE_DIR, train_class)
                if not os.path.isdir(train_dir):
                    continue

                train_images = sorted(glob.glob(os.path.join(train_dir, "*.png")))

                for train_img_path in train_images:
                    train_sig = extract_sig(train_img_path)
                    if train_sig is None:
                        continue

                    # Compute distance
                    dist = compute_bhattacharyya_distance(
                        failed_sig['histogram'],
                        train_sig['histogram']
                    )

                    best_matches.append({
                        'class': train_class,
                        'image': os.path.basename(train_img_path),
                        'distance': dist,
                        'path': train_img_path
                    })

            # Sort by distance (lower = more similar)
            best_matches.sort(key=lambda x: x['distance'])

            # Show top 5 matches
            print(f"\n  Top 5 most similar training images:")
            print(f"  {'Rank':<6} {'Class':<8} {'Image':<25} {'Distance':<12} {'Match?'}")
            print(f"  {'-' * 70}")

            for i, match in enumerate(best_matches[:5], 1):
                match_status = "✓ SAME" if match['class'] == test_class else "✗ DIFFERENT"
                marker = ""
                if match['class'] != test_class and match['distance'] < 0.15:
                    marker = " ⚠ POSSIBLE DATA LEAKAGE!"

                print(f"  {i:<6} {match['class']:<8} {match['image']:<25} {match['distance']:<12.4f} {match_status}{marker}")

            # Check if closest match is from a different class
            if best_matches[0]['class'] != test_class:
                print(f"\n  ⚠ WARNING: Closest match is from class {best_matches[0]['class']}, not class {test_class}!")
                print(f"    This test image might be mislabeled or there's data leakage in training data.")
                print(f"    Suggested action: Move {failed_img_name} from test/{test_class}/ to test/{best_matches[0]['class']}/")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\n1. Images marked with ⚠ POSSIBLE DATA LEAKAGE likely belong to a different class")
    print("2. Check if similar patterns exist in the wrong class's training folder")
    print("3. Move or remove mislabeled images from both train/ and test/ folders")
    print("4. Re-run train_all_templates.py after cleaning the data")
    print("5. Re-run test_all_images.py to verify improved accuracy")
