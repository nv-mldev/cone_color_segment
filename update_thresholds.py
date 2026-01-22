import os
import numpy as np
from utils.load_template import load_template
from utils.save_template import save_template

# ============================================================
# CONFIGURATION - SET YOUR OPTIMIZED THRESHOLDS HERE
# ============================================================

NEW_BHATT_THRESHOLD = 0.2736  # <-- Optimized value (95th percentile)
NEW_ENTROPY_THRESHOLD = 0.3647  # <-- Optimized value (95th percentile)

TEMPLATE_DIR = "templates"

# ============================================================
# UPDATE THRESHOLDS IN ALL TEMPLATES
# ============================================================

if __name__ == "__main__":
    print("Updating thresholds in all templates...")
    print(f"  New Bhattacharyya threshold: {NEW_BHATT_THRESHOLD}")
    print(f"  New Entropy threshold: {NEW_ENTROPY_THRESHOLD}")
    print("=" * 60)

    # Get all template files
    template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('_meta.json')]

    if not template_files:
        print(f"No templates found in {TEMPLATE_DIR}/")
        print("Run train_all_templates.py first!")
        exit()

    updated_count = 0

    for template_file in sorted(template_files):
        template_path = os.path.join(TEMPLATE_DIR, template_file.replace('_meta.json', ''))

        # Load template
        template = load_template(template_path)

        old_bhatt = template['bhatt_threshold']
        old_entropy = template['entropy_threshold']

        print(f"\n{template_file}:")
        print(f"  Pattern ID: {template['pattern_id']}")
        print(f"  Old Bhatt threshold: {old_bhatt:.4f} -> New: {NEW_BHATT_THRESHOLD:.4f}")
        print(f"  Old Entropy threshold: {old_entropy:.4f} -> New: {NEW_ENTROPY_THRESHOLD:.4f}")

        # Update thresholds
        template['bhatt_threshold'] = NEW_BHATT_THRESHOLD
        template['entropy_threshold'] = NEW_ENTROPY_THRESHOLD

        # Save updated template
        save_template(template, template_path)
        print(f"  ✓ Updated")

        updated_count += 1

    print("\n" + "=" * 60)
    print(f"Updated {updated_count} templates successfully!")
    print("\nYou can now run test_all_images.py to test with new thresholds.")
