import os

from utils.load_images import load_images_from_directory
from utils.train_template import train_template
from utils.save_template import save_template

# ============================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================

IMAGE_DIRECTORY = "/Users/nithinvadekkapat/work/cone_inspect/color_segment/data/9"  # <-- Change this path
PATTERN_ID = "P001_SAMPLE"  # <-- Change this pattern ID
OUTPUT_DIRECTORY = "/Users/nithinvadekkapat/work/cone_inspect/color_segment/templates"  # <-- Change this if needed


# ============================================================
# TRAINING SCRIPT
# ============================================================

if __name__ == "__main__":
    # Step 1: Load images
    print(f"Loading images from: {IMAGE_DIRECTORY}")
    images = load_images_from_directory(IMAGE_DIRECTORY)
    print(f"Loaded {len(images)} images")

    if len(images) < 2:
        print("Error: Need at least 2 images to train")
    else:
        # Step 2: Train template
        print(f"Training template for pattern: {PATTERN_ID}")
        template = train_template(images, PATTERN_ID)

        # Step 3: Print results
        print(f"\nTemplate trained successfully:")
        print(f"  Samples used: {template['sample_count']}")
        print(f"  Entropy: {template['entropy']:.4f}")
        print(f"  Mean L*: {template['mean_L']:.1f}")
        print(f"  Bhattacharyya threshold: {template['bhatt_threshold']}")
        print(f"  Entropy threshold: {template['entropy_threshold']}")

        # Step 4: Save template
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        output_base = os.path.join(OUTPUT_DIRECTORY, PATTERN_ID)
        save_template(template, output_base)
