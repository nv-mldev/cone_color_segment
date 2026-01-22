import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.load_images import load_images_from_directory
from utils.find_radius import find_radius
from utils.bilateral_filter import apply_bilateral_filter
from utils.convert_lab import convert_to_lab
from utils.unrolled import unroll_cone_tip
from utils.crop_sweet_spot import crop_polar_sweet_spot
from utils.histogram_2d import compute_2d_histogram
from utils.normalize_histogram import normalize_histogram_l1
from utils.entropy_2d import compute_2d_entropy
from utils.mean_lightness import compute_mean_lightness

# ============================================================
# CONFIGURATION - CHANGE THIS PATH
# ============================================================

IMAGE_PATH = "/Users/nithinvadekkapat/work/cone_inspect/color_segment/data/9/1481_vl.png"  # <-- Change this


# ============================================================
# VISUALIZATION SCRIPT
# ============================================================

if __name__ == "__main__":
    # Step 1: Load image
    print("Step 1: Loading image...")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load {IMAGE_PATH}")
        exit()
    print(f"  Image shape: {img.shape}")

    # Step 2: Find cone and get center/radius
    print("\nStep 2: Finding cone (center, radius)...")
    cropped_img, center, radius = find_radius(img)
    if cropped_img is None:
        print("Error: Could not find cone in image")
        exit()
    print(f"  Center: {center}")
    print(f"  Radius: {radius}")
    print(f"  Cropped shape: {cropped_img.shape}")

    # Step 3: Apply bilateral filter
    print("\nStep 3: Applying bilateral filter...")
    filtered = apply_bilateral_filter(cropped_img, d=9, sigma_color=30, sigma_space=9)
    print(f"  Filtered shape: {filtered.shape}")

    # Step 4: Convert to LAB
    print("\nStep 4: Converting to CIELAB...")
    lab = convert_to_lab(filtered)
    print(f"  LAB shape: {lab.shape}")

    # Step 5: Polar warp (unroll)
    print("\nStep 5: Polar warping (unrolling circle)...")
    polar = unroll_cone_tip(lab, center, radius)
    print(f"  Polar shape: {polar.shape}")

    # Step 6: Crop sweet spot (remove black regions + edges)
    print("\nStep 6: Cropping sweet spot...")
    lab_patch = crop_polar_sweet_spot(
        polar, inner_crop_pct=0.10, outer_crop_pct=0.10, debug=True
    )

    # Step 7: Compute 2D histogram
    print("\nStep 7: Computing 2D histogram (a* vs b*)...")
    hist = compute_2d_histogram(lab_patch)
    print(f"  Histogram shape: {hist.shape}")

    # Step 8: Normalize histogram
    print("\nStep 8: L1 normalizing histogram...")
    hist_norm = normalize_histogram_l1(hist)
    print(f"  Sum after normalization: {hist_norm.sum():.6f}")

    # Step 9: Compute entropy
    print("\nStep 9: Computing 2D entropy...")
    entropy = compute_2d_entropy(hist_norm)
    print(f"  Entropy: {entropy:.4f}")

    # Step 10: Compute mean lightness
    print("\nStep 10: Computing mean L*...")
    mean_L = compute_mean_lightness(lab_patch)
    print(f"  Mean L*: {mean_L:.2f}")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\n" + "=" * 50)
    print("Generating visualization...")
    print("=" * 50)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Row 1: Original -> Cropped -> Filtered
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"2. Cropped (r={radius})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("3. Bilateral Filtered")
    axes[0, 2].axis("off")

    # Row 2: LAB channels -> Polar -> Cropped patch
    # Show LAB as RGB for visualization (it won't look correct but shows the data)
    axes[1, 0].imshow(lab[:, :, 0], cmap="gray")
    axes[1, 0].set_title("4. L* channel (Lightness)")
    axes[1, 0].axis("off")

    # Convert polar LAB back to BGR for display
    polar_bgr = cv2.cvtColor(polar, cv2.COLOR_LAB2BGR)
    axes[1, 1].imshow(cv2.cvtColor(polar_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. Polar Warped")
    axes[1, 1].axis("off")

    # Convert cropped patch LAB back to BGR for display
    patch_bgr = cv2.cvtColor(lab_patch, cv2.COLOR_LAB2BGR)
    axes[1, 2].imshow(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Sweet Spot (Final Patch)")
    axes[1, 2].axis("off")

    # Row 3: a* channel, b* channel, 2D histogram
    axes[2, 0].imshow(lab_patch[:, :, 1], cmap="RdYlGn_r")
    axes[2, 0].set_title("7. a* channel (Green-Red)")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(lab_patch[:, :, 2], cmap="YlGnBu_r")
    axes[2, 1].set_title("8. b* channel (Blue-Yellow)")
    axes[2, 1].axis("off")

    # 2D Histogram
    im = axes[2, 2].imshow(hist_norm, cmap="hot", origin="lower", aspect="auto")
    axes[2, 2].set_title(f"9. 2D Histogram (Entropy={entropy:.2f})")
    axes[2, 2].set_xlabel("b* bins")
    axes[2, 2].set_ylabel("a* bins")
    plt.colorbar(im, ax=axes[2, 2], fraction=0.046)

    plt.suptitle(
        f"Pipeline Visualization | Mean L*: {mean_L:.1f} | Entropy: {entropy:.2f}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    print("\nDone!")
