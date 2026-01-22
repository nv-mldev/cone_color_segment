# Cone Color Segmentation - Template Training & Testing

## Overview

This project provides an automated pipeline for training templates from cone images and testing them against new images. The system uses color histograms, entropy, and Bhattacharyya distance for pattern matching.

## Workflow

### 1. Train Templates for All Classes

```bash
python train_all_templates.py
```

**What it does:**
- Scans all subdirectories in `train/` folder (classes 1, 2, 3, 5, 6, 9, 10)
- For each class:
  - Loads all training images
  - Computes average color histogram (a*, b* channels in LAB color space)
  - Calculates reference entropy and mean lightness
  - Sets default thresholds
  - Saves template to `templates/class_X.npz`

**Output:**
- Templates saved in `templates/` folder
- Summary table showing entropy, mean L*, and thresholds for each class

### 2. Test All Images Against Templates

```bash
python test_all_images.py
```

**What it does:**
- For each class in `test/` folder:
  - Loads corresponding template from `templates/`
  - Tests each image against its template
  - Computes Bhattacharyya distance and entropy delta
  - Determines PASS/FAIL based on thresholds
- Generates accuracy report per class and overall

**Output:**
- Console output showing PASS/FAIL for each image
- Accuracy summary per class
- Detailed results saved to `test_results.json`

### 3. Optimize Thresholds

```bash
python optimize_thresholds.py
```

**What it does:**
- Computes **intra-class** metrics (same class images vs their template)
- Computes **inter-class** metrics (images vs wrong templates)
- Analyzes the separation between same-class and different-class matches
- Recommends optimal thresholds using 3 methods:
  1. Mean + 2*Std (statistical approach)
  2. 95th percentile (covers 95% of valid samples)
  3. Midpoint between intra-max and inter-min (balanced approach)

**Output:**
- Statistical analysis printed to console
- Recommended thresholds for Bhattacharyya distance and entropy
- Visualization saved to `threshold_analysis.png` showing:
  - Distribution histograms (intra vs inter class)
  - Box plots for both metrics
  - Suggested threshold lines

## Understanding Thresholds

### Bhattacharyya Distance
- Measures similarity between two histograms
- **Lower values** = more similar (GOOD for matching)
- **Higher values** = less similar (BAD for matching)
- Typical range: 0.0 to 1.0
- **Threshold**: Images with distance BELOW threshold are considered a match

### Entropy Delta
- Measures the difference in color diversity
- **Lower values** = similar complexity (GOOD for matching)
- **Higher values** = different complexity (BAD for matching)
- **Threshold**: Images with delta BELOW threshold are considered a match

### Choosing the Right Threshold

1. **Conservative (fewer false positives)**: Use Method 3 (Midpoint)
   - Safer for quality control
   - May reject some valid samples

2. **Balanced (good tradeoff)**: Use Method 2 (95th percentile)
   - Recommended starting point
   - Covers 95% of valid intra-class variation

3. **Aggressive (fewer false negatives)**: Use Method 1 (Mean + 2*Std)
   - May allow some outliers to pass
   - Better for maximizing yield

## How to Apply New Thresholds

After running `optimize_thresholds.py`, you'll get recommended values like:

```
Bhattacharyya threshold: 0.1234
Entropy threshold: 0.5678
```

To apply these thresholds, you need to update the template training code. The thresholds are set in:

**Option 1: Modify `utils/train_template.py`**
- Look for where `bhatt_threshold` and `entropy_threshold` are set
- Update with your optimized values

**Option 2: Retrain with custom thresholds**
- Modify `train_all_templates.py` to override thresholds after training
- Example:
  ```python
  template = train_template(images, pattern_id)
  template['bhatt_threshold'] = 0.1234  # Your optimized value
  template['entropy_threshold'] = 0.5678  # Your optimized value
  save_template(template, output_base)
  ```

Then re-run the training and testing:
```bash
python train_all_templates.py
python test_all_images.py
```

## Expected Results

Good separability:
- Intra-class metrics should be **low** (similar to template)
- Inter-class metrics should be **high** (different from template)
- Clear gap between distributions

If there's overlap:
- Consider improving image preprocessing
- Check if classes are truly distinct
- May need to adjust bilateral filter or cropping parameters

## Troubleshooting

**Error: Template not found**
- Run `train_all_templates.py` first

**Low accuracy on test images**
- Check threshold analysis visualization
- May need to optimize thresholds
- Verify training images are representative

**All images failing**
- Thresholds may be too strict
- Check if preprocessing parameters match between train and test

**All images passing (even wrong classes)**
- Thresholds may be too loose
- Run `optimize_thresholds.py` to find better values
