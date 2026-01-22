import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# VISUALIZE FAILED IMAGES
# ============================================================

RESULTS_FILE = "test_results.json"
TEST_BASE_DIR = "test"

if __name__ == "__main__":
    # Load test results
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    # Filter failed images
    failed_images = [r for r in results['detailed_results'] if r['status'] == 'FAIL']

    if not failed_images:
        print("No failed images found!")
        exit()

    print(f"Visualizing {len(failed_images)} failed images...")

    # Calculate grid size
    n_images = len(failed_images)
    cols = 4
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, img_data in enumerate(failed_images):
        class_id = img_data['class']
        img_name = img_data['image']
        img_path = os.path.join(TEST_BASE_DIR, class_id, img_name)

        # Load image
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img_rgb)
        else:
            axes[idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')

        # Title with details
        title = f"Class {class_id}: {img_name}\n"
        title += f"Bhatt={img_data['bhatt_distance']:.4f}, "
        title += f"Ent Δ={img_data['entropy_delta']:.4f}\n"
        title += f"Conf={img_data['confidence']}%"
        if img_data['illumination_warning']:
            title += " [ILLUM WARN]"

        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('failed_images.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: failed_images.png")
    plt.show()

    print("\n" + "=" * 80)
    print("SUMMARY OF FAILURES")
    print("=" * 80)

    # Group by class
    failures_by_class = {}
    for img in failed_images:
        class_id = img['class']
        if class_id not in failures_by_class:
            failures_by_class[class_id] = []
        failures_by_class[class_id].append(img['image'])

    for class_id in sorted(failures_by_class.keys()):
        print(f"\nClass {class_id}: {failures_by_class[class_id]}")
