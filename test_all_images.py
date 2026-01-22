import os
import cv2
import glob
import json
from utils.load_template import load_template
from utils.extract_signature import extract_signature_from_image
from utils.match_pattern import match_pattern

# ============================================================
# CONFIGURATION
# ============================================================

TEST_BASE_DIR = "test"
TEMPLATE_DIR = "templates"
RESULTS_FILE = "test_results.json"

# ============================================================
# TEST ALL IMAGES
# ============================================================

if __name__ == "__main__":
    # Get all class directories in test folder
    class_dirs = sorted([d for d in os.listdir(TEST_BASE_DIR)
                        if os.path.isdir(os.path.join(TEST_BASE_DIR, d))])

    print(f"Found {len(class_dirs)} test classes: {class_dirs}")
    print("=" * 80)

    all_results = []
    class_summary = {}

    for class_id in class_dirs:
        test_dir = os.path.join(TEST_BASE_DIR, class_id)
        template_path = os.path.join(TEMPLATE_DIR, f"class_{class_id}")

        print(f"\n[Class {class_id}] Testing images...")

        # Check if template exists
        if not os.path.exists(f"{template_path}_meta.json"):
            print(f"  ERROR: Template not found at {template_path}_meta.json")
            print(f"  Run train_all_templates.py first!")
            continue

        # Load template
        template = load_template(template_path)
        print(f"  Template loaded: {template['pattern_id']}")
        print(f"  Bhatt threshold: {template['bhatt_threshold']:.4f}")
        print(f"  Entropy threshold: {template['entropy_threshold']:.4f}")

        # Get all test images
        test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
        print(f"  Found {len(test_images)} test images")

        # Track results for this class
        class_results = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'details': []
        }

        # Test each image
        for img_path in test_images:
            img_name = os.path.basename(img_path)

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"    {img_name}: ERROR - Could not load image")
                class_results['errors'] += 1
                continue

            # Extract signature
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
                print(f"    {img_name}: ERROR - Could not extract signature")
                class_results['errors'] += 1
                continue

            # Match against template
            result = match_pattern(
                live_hist=sig['histogram'],
                live_entropy=sig['entropy'],
                live_L=sig['mean_L'],
                master_hist=template['histogram'],
                master_entropy=template['entropy'],
                master_L=template['mean_L'],
                bhatt_threshold=template['bhatt_threshold'],
                entropy_threshold=template['entropy_threshold']
            )

            # Track result
            status = "PASS" if result['pass'] else "FAIL"
            if result['pass']:
                class_results['passed'] += 1
            else:
                class_results['failed'] += 1

            # Store detailed result
            detail = {
                'image': img_name,
                'class': class_id,
                'status': status,
                'confidence': result['confidence'],
                'bhatt_distance': result['bhattacharyya_distance'],
                'entropy_delta': result['entropy_delta'],
                'L_drift_percent': result['L_drift_percent'],
                'illumination_warning': result['illumination_warning']
            }
            class_results['details'].append(detail)
            all_results.append(detail)

            # Print result
            warning = " [ILLUMINATION WARNING]" if result['illumination_warning'] else ""
            print(f"    {img_name}: {status} (conf={result['confidence']}%, bhatt={result['bhattacharyya_distance']:.4f}){warning}")

        # Print class summary
        total = class_results['passed'] + class_results['failed']
        accuracy = (class_results['passed'] / total * 100) if total > 0 else 0

        print(f"\n  Class {class_id} Summary:")
        print(f"    Passed: {class_results['passed']}/{total} ({accuracy:.1f}%)")
        print(f"    Failed: {class_results['failed']}/{total}")
        print(f"    Errors: {class_results['errors']}")

        class_summary[class_id] = {
            'passed': class_results['passed'],
            'failed': class_results['failed'],
            'errors': class_results['errors'],
            'total': total,
            'accuracy': accuracy
        }

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"{'Class':<8} {'Passed':<10} {'Failed':<10} {'Errors':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 80)

    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_images = 0

    for class_id in sorted(class_summary.keys()):
        s = class_summary[class_id]
        print(f"{class_id:<8} {s['passed']:<10} {s['failed']:<10} {s['errors']:<10} {s['total']:<10} {s['accuracy']:<10.1f}%")
        total_passed += s['passed']
        total_failed += s['failed']
        total_errors += s['errors']
        total_images += s['total']

    print("-" * 80)
    overall_accuracy = (total_passed / total_images * 100) if total_images > 0 else 0
    print(f"{'TOTAL':<8} {total_passed:<10} {total_failed:<10} {total_errors:<10} {total_images:<10} {overall_accuracy:<10.1f}%")

    # Save detailed results to JSON
    output_data = {
        'class_summary': class_summary,
        'total_passed': total_passed,
        'total_failed': total_failed,
        'total_errors': total_errors,
        'total_images': total_images,
        'overall_accuracy': overall_accuracy,
        'detailed_results': all_results
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {RESULTS_FILE}")
    print("Done!")
