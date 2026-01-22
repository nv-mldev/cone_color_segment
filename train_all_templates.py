import os
import glob
from utils.load_images import load_images_from_directory
from utils.train_template import train_template
from utils.save_template import save_template

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_BASE_DIR = "train"
OUTPUT_DIRECTORY = "templates"

# ============================================================
# TRAIN ALL TEMPLATES
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(TRAIN_BASE_DIR)
                        if os.path.isdir(os.path.join(TRAIN_BASE_DIR, d))])

    print(f"Found {len(class_dirs)} classes to train: {class_dirs}")
    print("=" * 60)

    results = []

    for class_id in class_dirs:
        class_dir = os.path.join(TRAIN_BASE_DIR, class_id)
        pattern_id = f"CONE_CLASS_{class_id}"

        print(f"\n[{class_id}] Training template for class {class_id}...")
        print(f"  Directory: {class_dir}")

        # Load images
        images = load_images_from_directory(class_dir)
        print(f"  Loaded {len(images)} images")

        if len(images) < 2:
            print(f"  ERROR: Need at least 2 images, found {len(images)}")
            continue

        # Train template
        template = train_template(images, pattern_id)

        # Print results
        print(f"  Template trained successfully:")
        print(f"    Samples used: {template['sample_count']}")
        print(f"    Entropy: {template['entropy']:.4f}")
        print(f"    Mean L*: {template['mean_L']:.1f}")
        print(f"    Bhattacharyya threshold: {template['bhatt_threshold']:.4f}")
        print(f"    Entropy threshold: {template['entropy_threshold']:.4f}")

        # Save template
        output_base = os.path.join(OUTPUT_DIRECTORY, f"class_{class_id}")
        save_template(template, output_base)
        print(f"  Saved to: {output_base}")

        results.append({
            'class_id': class_id,
            'pattern_id': pattern_id,
            'sample_count': template['sample_count'],
            'entropy': template['entropy'],
            'mean_L': template['mean_L'],
            'bhatt_threshold': template['bhatt_threshold'],
            'entropy_threshold': template['entropy_threshold']
        })

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Class':<8} {'Samples':<10} {'Entropy':<12} {'Mean L*':<10} {'Bhatt Thr':<12} {'Ent Thr':<12}")
    print("-" * 60)

    for r in results:
        print(f"{r['class_id']:<8} {r['sample_count']:<10} {r['entropy']:<12.4f} {r['mean_L']:<10.1f} {r['bhatt_threshold']:<12.4f} {r['entropy_threshold']:<12.4f}")

    print(f"\nAll templates saved to: {OUTPUT_DIRECTORY}/")
    print("Done!")
