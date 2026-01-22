import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# VISUALIZE CLASS 9 PROBLEM IMAGES
# ============================================================

# Outlier training images
OUTLIER_TRAIN = [
    'train/9/3439_vl.png',
    'train/9/4640_vl.png'
]

# Normal training images
NORMAL_TRAIN = [
    'train/9/1481_vl.png',
    'train/9/5185_vl.png',
    'train/9/149_vl.png',
    'train/9/2635_vl.png'
]

# Failed test images (match the outliers)
FAILED_TEST = [
    'test/9/7981_vl.png',
    'test/9/8341_vl.png',
    'test/9/9416_vl.png',
    'test/9/9951_vl.png'
]

# Passed test images (match normal pattern)
PASSED_TEST = [
    'test/9/7124_vl.png',
    'test/9/8086_vl.png',
    'test/9/8760_vl.png',
    'test/9/9360_vl.png'
]


if __name__ == "__main__":
    print("Visualizing Class 9 patterns...")

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    # Row 0: Outlier training images (Pattern B)
    for i, img_path in enumerate(OUTLIER_TRAIN):
        img = cv2.imread(img_path)
        if img is not None:
            axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"OUTLIER TRAIN\n{os.path.basename(img_path)}", fontsize=9, color='red')
        axes[0, i].axis('off')

    # Fill remaining row 0 cells
    for i in range(len(OUTLIER_TRAIN), 4):
        axes[0, i].axis('off')

    # Row 1: Normal training images (Pattern A)
    for i, img_path in enumerate(NORMAL_TRAIN):
        img = cv2.imread(img_path)
        if img is not None:
            axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"NORMAL TRAIN\n{os.path.basename(img_path)}", fontsize=9, color='green')
        axes[1, i].axis('off')

    # Row 2: Failed test images (match outliers)
    for i, img_path in enumerate(FAILED_TEST):
        img = cv2.imread(img_path)
        if img is not None:
            axes[2, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[2, i].set_title(f"FAILED TEST\n{os.path.basename(img_path)}\n(matches outliers)", fontsize=9, color='red')
        axes[2, i].axis('off')

    # Row 3: Passed test images (match normal pattern)
    for i, img_path in enumerate(PASSED_TEST):
        img = cv2.imread(img_path)
        if img is not None:
            axes[3, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[3, i].set_title(f"PASSED TEST\n{os.path.basename(img_path)}\n(matches normal)", fontsize=9, color='green')
        axes[3, i].axis('off')

    plt.suptitle('Class 9: Two Different Patterns Detected', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('class9_pattern_comparison.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: class9_pattern_comparison.png")
    plt.show()

    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)
    print("\nRow 0 (RED): Outlier training images - These are VERY different from normal class 9")
    print("Row 1 (GREEN): Normal training images - These are the majority class 9 pattern")
    print("Row 2 (RED): Failed test images - These match the outlier pattern, not the template")
    print("Row 3 (GREEN): Passed test images - These match the normal pattern")
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("\nLook at the images and decide:")
    print("1. Are RED images (outliers) the WRONG pattern? → Remove them")
    print("2. Are RED images a valid but DIFFERENT class? → Create new class")
    print("3. Are GREEN images actually wrong? → (unlikely based on data)")
    print("\nMost likely: RED images are mislabeled or single-color/defect cones")
