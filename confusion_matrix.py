import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils.load_template import load_template
from utils.extract_signature import extract_signature_from_image
from utils.bhattacharyya_distance import compute_bhattacharyya_distance

# ============================================================
# CONFUSION MATRIX - TEST ALL CLASSES AGAINST ALL TEMPLATES
# ============================================================

TEST_BASE_DIR = "test"
TEMPLATE_DIR = "templates"

if __name__ == "__main__":
    # Get all classes
    class_dirs = sorted([d for d in os.listdir(TEST_BASE_DIR)
                        if os.path.isdir(os.path.join(TEST_BASE_DIR, d))])

    print("=" * 80)
    print("GENERATING CONFUSION MATRIX")
    print("=" * 80)
    print(f"\nTesting {len(class_dirs)} classes against all templates...\n")

    # Load all templates
    templates = {}
    for class_id in class_dirs:
        template_path = os.path.join(TEMPLATE_DIR, f"class_{class_id}")
        if os.path.exists(f"{template_path}_meta.json"):
            templates[class_id] = load_template(template_path)
            print(f"Loaded template for class {class_id}")

    print(f"\nLoaded {len(templates)} templates\n")

    # Initialize confusion matrix
    # Rows = True class, Columns = Predicted class
    confusion = np.zeros((len(class_dirs), len(class_dirs)))
    class_to_idx = {c: i for i, c in enumerate(class_dirs)}

    # Store detailed results
    detailed_results = []

    # Test each class
    for true_class in class_dirs:
        test_dir = os.path.join(TEST_BASE_DIR, true_class)
        test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))

        print(f"Testing class {true_class}: {len(test_images)} images")

        for img_path in test_images:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Extract signature
            sig = extract_signature_from_image(img)
            if sig is None:
                continue

            # Test against ALL templates
            best_class = None
            best_distance = float('inf')
            all_distances = {}

            for template_class, template in templates.items():
                params = template['preprocess_params']

                # Re-extract with template's params (in case they differ)
                sig_test = extract_signature_from_image(
                    img,
                    inner_crop_pct=params['inner_crop_pct'],
                    outer_crop_pct=params['outer_crop_pct'],
                    bilateral_d=params['bilateral_d'],
                    bilateral_sigma_color=params['bilateral_sigma_color'],
                    bilateral_sigma_space=params['bilateral_sigma_space']
                )

                if sig_test is None:
                    continue

                # Compute distance
                dist = compute_bhattacharyya_distance(
                    sig_test['histogram'],
                    template['histogram']
                )

                all_distances[template_class] = dist

                # Track best match
                if dist < best_distance:
                    best_distance = dist
                    best_class = template_class

            # Update confusion matrix
            true_idx = class_to_idx[true_class]
            pred_idx = class_to_idx[best_class] if best_class else true_idx

            confusion[true_idx, pred_idx] += 1

            # Store result
            detailed_results.append({
                'true_class': true_class,
                'predicted_class': best_class,
                'image': img_name,
                'best_distance': best_distance,
                'all_distances': all_distances,
                'correct': (true_class == best_class)
            })

    # Print confusion matrix
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (Rows=True Class, Columns=Predicted Class)")
    print("=" * 80)

    # Header
    header = "True\\Pred"
    for c in class_dirs:
        header += f" | {c:>4}"
    header += " | Total | Acc%"
    print(header)
    print("-" * len(header))

    # Rows
    for i, true_class in enumerate(class_dirs):
        row = f"{true_class:>9}"
        row_total = 0
        correct = 0

        for j, pred_class in enumerate(class_dirs):
            count = int(confusion[i, j])
            row += f" | {count:>4}"
            row_total += count
            if i == j:
                correct = count

        accuracy = (correct / row_total * 100) if row_total > 0 else 0
        row += f" | {row_total:>5} | {accuracy:>4.1f}"
        print(row)

    # Overall accuracy
    total_correct = np.trace(confusion)
    total_samples = np.sum(confusion)
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0

    print("-" * len(header))
    print(f"Overall Accuracy: {total_correct:.0f}/{total_samples:.0f} = {overall_accuracy:.2f}%")

    # Per-class accuracy
    print("\n" + "=" * 80)
    print("PER-CLASS ACCURACY")
    print("=" * 80)
    print(f"{'Class':<8} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Errors'}")
    print("-" * 80)

    for i, class_id in enumerate(class_dirs):
        correct = int(confusion[i, i])
        total = int(np.sum(confusion[i, :]))
        accuracy = (correct / total * 100) if total > 0 else 0

        # Find where errors went
        errors = []
        for j, other_class in enumerate(class_dirs):
            if i != j and confusion[i, j] > 0:
                errors.append(f"{int(confusion[i, j])} → class {other_class}")

        error_str = ", ".join(errors) if errors else "None"

        print(f"{class_id:<8} {correct:<10} {total:<10} {accuracy:<10.1f}% {error_str}")

    # Misclassifications detail
    print("\n" + "=" * 80)
    print("MISCLASSIFIED IMAGES")
    print("=" * 80)

    misclassified = [r for r in detailed_results if not r['correct']]

    if misclassified:
        print(f"\nTotal misclassified: {len(misclassified)}\n")

        for result in misclassified:
            print(f"{result['image']} (true: {result['true_class']}, predicted: {result['predicted_class']})")
            print(f"  Best distance: {result['best_distance']:.4f}")

            # Show top 3 closest matches
            sorted_dists = sorted(result['all_distances'].items(), key=lambda x: x[1])
            print(f"  Top 3 matches:")
            for rank, (cls, dist) in enumerate(sorted_dists[:3], 1):
                marker = "✓" if cls == result['true_class'] else "✗"
                print(f"    {rank}. Class {cls}: {dist:.4f} {marker}")
            print()
    else:
        print("\nNo misclassifications! Perfect accuracy!\n")

    # Visualize confusion matrix
    print("Generating visualization...")

    # Normalize confusion matrix for heatmap (percentage)
    confusion_pct = np.zeros_like(confusion)
    for i in range(len(class_dirs)):
        row_sum = np.sum(confusion[i, :])
        if row_sum > 0:
            confusion_pct[i, :] = confusion[i, :] / row_sum * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    im1 = ax1.imshow(confusion, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(len(class_dirs)))
    ax1.set_yticks(range(len(class_dirs)))
    ax1.set_xticklabels(class_dirs)
    ax1.set_yticklabels(class_dirs)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('True Class', fontsize=12)

    # Annotate counts
    for i in range(len(class_dirs)):
        for j in range(len(class_dirs)):
            text = ax1.text(j, i, f'{int(confusion[i, j])}',
                           ha="center", va="center", color="black" if confusion[i, j] < confusion.max()/2 else "white")

    plt.colorbar(im1, ax=ax1, label='Count')

    # Percentage
    im2 = ax2.imshow(confusion_pct, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(range(len(class_dirs)))
    ax2.set_yticks(range(len(class_dirs)))
    ax2.set_xticklabels(class_dirs)
    ax2.set_yticklabels(class_dirs)
    ax2.set_title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Class', fontsize=12)
    ax2.set_ylabel('True Class', fontsize=12)

    # Annotate percentages
    for i in range(len(class_dirs)):
        for j in range(len(class_dirs)):
            text = ax2.text(j, i, f'{confusion_pct[i, j]:.1f}',
                           ha="center", va="center", color="black" if confusion_pct[i, j] < 50 else "white")

    plt.colorbar(im2, ax=ax2, label='Percentage (%)')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved to: confusion_matrix.png")

    plt.show()

    print("\nDone!")
