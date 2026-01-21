import os
import cv2

from utils.load_template import load_template
from utils.extract_signature import extract_signature_from_image
from utils.match_pattern import match_pattern

# ============================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================

TEMPLATE_PATH = "/Users/nithinvadekkapat/work/cone_inspect/color_segment/templates/P001_SAMPLE"  # <-- Without extension
IMAGE_PATH = "/Users/nithinvadekkapat/work/cone_inspect/color_segment/data/9/1481_vl.png"  # <-- Test image


# ============================================================
# INFERENCE SCRIPT
# ============================================================

if __name__ == "__main__":
    # Step 1: Load template
    print(f"Loading template from: {TEMPLATE_PATH}")
    template = load_template(TEMPLATE_PATH)
    print(f"  Pattern ID: {template['pattern_id']}")
    print(f"  Bhatt threshold: {template['bhatt_threshold']}")
    print(f"  Entropy threshold: {template['entropy_threshold']}")

    # Step 2: Load and process test image
    print(f"\nProcessing image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load {IMAGE_PATH}")
        exit()

    # Step 3: Extract signature from live image
    print("Extracting signature...")
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
        print("Error: Could not extract signature from image")
        exit()

    # Step 4: Match against template using Bhattacharyya distance
    print("Matching against template...")
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

    # Step 5: Print results
    print("\n" + "="*50)
    print("RESULT")
    print("="*50)
    print(f"  Status: {'PASS' if result['pass'] else 'FAIL'}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Bhattacharyya distance: {result['bhattacharyya_distance']} (threshold: {template['bhatt_threshold']})")
    print(f"  Entropy delta: {result['entropy_delta']} (threshold: {template['entropy_threshold']})")
    if result['illumination_warning']:
        print(f"  WARNING: Illumination drift {result['L_drift_percent']}%")
