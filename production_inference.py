import cv2
import numpy as np
from utils.load_template import load_template
from utils.extract_signature import extract_signature_from_image
from utils.bhattacharyya_distance import compute_bhattacharyya_distance

# ============================================================
# PRODUCTION INFERENCE - HYBRID APPROACH
# ============================================================

class ConeClassifier:
    """
    Production-ready cone classifier with hybrid verification.

    Combines threshold-based quality control with nearest-neighbor
    mislabel detection.
    """

    def __init__(self, template_dir="templates"):
        """
        Initialize classifier by loading ALL templates at startup.

        Memory usage: ~7 templates × 4KB = ~28KB (tiny!)
        """
        self.templates = {}
        self.class_ids = ['1', '2', '3', '5', '6', '9', '10']

        print("Loading templates...")
        for class_id in self.class_ids:
            template_path = f"{template_dir}/class_{class_id}"
            self.templates[class_id] = load_template(template_path)
            print(f"  ✓ Loaded template for class {class_id}")

        print(f"\nLoaded {len(self.templates)} templates successfully!")
        print(f"Memory usage: ~{len(self.templates) * 4}KB (negligible)\n")


    def verify_cone(self, image, expected_class_id):
        """
        Verify a cone against expected class from PLC.

        Args:
            image: OpenCV image (BGR)
            expected_class_id: Class ID from PLC (e.g., "9")

        Returns:
            dict with verification results
        """
        # Extract signature
        sig = extract_signature_from_image(image)
        if sig is None:
            return {
                'status': 'ERROR',
                'error': 'Could not extract cone signature',
                'pass': False
            }

        # Get expected template
        if expected_class_id not in self.templates:
            return {
                'status': 'ERROR',
                'error': f'Unknown class ID: {expected_class_id}',
                'pass': False
            }

        expected_template = self.templates[expected_class_id]

        # 1. THRESHOLD CHECK - Quality Control
        expected_distance = compute_bhattacharyya_distance(
            sig['histogram'],
            expected_template['histogram']
        )

        threshold_pass = expected_distance < expected_template['bhatt_threshold']

        # 2. NEAREST NEIGHBOR - Mislabel Detection
        distances = {}
        for class_id, template in self.templates.items():
            dist = compute_bhattacharyya_distance(
                sig['histogram'],
                template['histogram']
            )
            distances[class_id] = dist

        # Find closest match
        predicted_class = min(distances, key=distances.get)
        closest_distance = distances[predicted_class]

        # 3. DECISION LOGIC
        # Case 1: Perfect match (threshold pass + correct prediction)
        if threshold_pass and predicted_class == expected_class_id:
            status = 'PASS'
            warning = None

        # Case 2: Outlier (threshold fail but still closest to expected)
        elif not threshold_pass and predicted_class == expected_class_id:
            status = 'PASS_WITH_WARNING'
            warning = f'Outlier detected (distance={expected_distance:.4f} > threshold={expected_template["bhatt_threshold"]:.4f})'

        # Case 3: MISLABELED! (closest to different class)
        elif predicted_class != expected_class_id:
            status = 'FAIL_MISLABEL'
            warning = f'MISLABEL DETECTED! Expected class {expected_class_id}, but closest to class {predicted_class}'

        # Case 4: Threshold fail and mislabeled
        else:
            status = 'FAIL'
            warning = 'Both threshold and prediction failed'

        # Compute confidence
        confidence = max(0, min(100, (1 - expected_distance / expected_template['bhatt_threshold']) * 100))

        return {
            'status': status,
            'pass': status.startswith('PASS'),
            'expected_class': expected_class_id,
            'predicted_class': predicted_class,
            'distance_to_expected': expected_distance,
            'distance_to_closest': closest_distance,
            'threshold': expected_template['bhatt_threshold'],
            'confidence': round(confidence, 1),
            'warning': warning,
            'all_distances': distances,
            'entropy': sig['entropy'],
            'mean_L': sig['mean_L']
        }


    def classify_unknown(self, image):
        """
        Classify an unknown cone (when PLC class ID is not available).

        Args:
            image: OpenCV image (BGR)

        Returns:
            dict with classification results
        """
        # Extract signature
        sig = extract_signature_from_image(image)
        if sig is None:
            return {
                'status': 'ERROR',
                'error': 'Could not extract cone signature',
                'predicted_class': None
            }

        # Find nearest neighbor
        distances = {}
        for class_id, template in self.templates.items():
            dist = compute_bhattacharyya_distance(
                sig['histogram'],
                template['histogram']
            )
            distances[class_id] = dist

        # Get prediction
        predicted_class = min(distances, key=distances.get)
        predicted_distance = distances[predicted_class]
        predicted_template = self.templates[predicted_class]

        # Check if within threshold
        within_threshold = predicted_distance < predicted_template['bhatt_threshold']

        # Compute confidence
        confidence = max(0, min(100, (1 - predicted_distance / predicted_template['bhatt_threshold']) * 100))

        # Get second best for comparison
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        second_best_class = sorted_distances[1][0] if len(sorted_distances) > 1 else None
        second_best_distance = sorted_distances[1][1] if len(sorted_distances) > 1 else None

        # Calculate separation margin
        margin = second_best_distance - predicted_distance if second_best_distance else None

        return {
            'status': 'CLASSIFIED',
            'predicted_class': predicted_class,
            'confidence': round(confidence, 1),
            'distance': predicted_distance,
            'threshold': predicted_template['bhatt_threshold'],
            'within_threshold': within_threshold,
            'second_best_class': second_best_class,
            'separation_margin': round(margin, 4) if margin else None,
            'all_distances': distances,
            'entropy': sig['entropy'],
            'mean_L': sig['mean_L']
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Initialize classifier (load all templates once at startup)
    classifier = ConeClassifier()

    print("=" * 80)
    print("EXAMPLE 1: VERIFY CONE WITH PLC CLASS ID")
    print("=" * 80)

    # Simulate: PLC says this should be class 9
    plc_class_id = "9"
    test_image_path = "test/9/8760_vl.png"

    print(f"\nPLC Class ID: {plc_class_id}")
    print(f"Test Image: {test_image_path}")

    # Load image
    image = cv2.imread(test_image_path)

    # Verify
    result = classifier.verify_cone(image, plc_class_id)

    print(f"\nResult:")
    print(f"  Status: {result['status']}")
    print(f"  Pass: {result['pass']}")
    print(f"  Expected: Class {result['expected_class']}")
    print(f"  Predicted: Class {result['predicted_class']}")
    print(f"  Distance: {result['distance_to_expected']:.4f} (threshold: {result['threshold']:.4f})")
    print(f"  Confidence: {result['confidence']}%")
    if result['warning']:
        print(f"  ⚠️ Warning: {result['warning']}")

    print("\n" + "=" * 80)
    print("EXAMPLE 2: CLASSIFY UNKNOWN CONE (NO PLC DATA)")
    print("=" * 80)

    test_image_path = "test/2/3725_vl.png"
    print(f"\nTest Image: {test_image_path}")

    image = cv2.imread(test_image_path)
    result = classifier.classify_unknown(image)

    print(f"\nResult:")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Distance: {result['distance']:.4f} (threshold: {result['threshold']:.4f})")
    print(f"  Within Threshold: {result['within_threshold']}")
    print(f"  Second Best: Class {result['second_best_class']} (margin: {result['separation_margin']})")

    print("\n" + "=" * 80)
    print("EXAMPLE 3: MISLABEL DETECTION")
    print("=" * 80)

    # Simulate mislabel: PLC says class 5, but it's actually class 2
    plc_class_id = "5"
    test_image_path = "test/2/3725_vl.png"

    print(f"\nPLC Class ID: {plc_class_id} (WRONG!)")
    print(f"Test Image: {test_image_path} (actually class 2)")

    image = cv2.imread(test_image_path)
    result = classifier.verify_cone(image, plc_class_id)

    print(f"\nResult:")
    print(f"  Status: {result['status']}")
    print(f"  Pass: {result['pass']}")
    print(f"  Expected: Class {result['expected_class']}")
    print(f"  Predicted: Class {result['predicted_class']}")
    print(f"  🚨 {result['warning']}")

    print("\n" + "=" * 80)
    print("ALL DISTANCES FOR VERIFICATION:")
    print("=" * 80)
    for class_id, dist in sorted(result['all_distances'].items(), key=lambda x: x[1]):
        marker = "✓ CLOSEST" if class_id == result['predicted_class'] else ""
        marker += " ← PLC" if class_id == result['expected_class'] else ""
        print(f"  Class {class_id}: {dist:.4f} {marker}")

    print("\nDone!")
