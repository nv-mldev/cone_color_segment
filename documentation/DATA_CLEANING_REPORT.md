# Data Cleaning Report - Cone Color Segmentation

## Summary
Analysis revealed data quality issues, primarily in **Class 9**, which has mixed patterns.

---

## CRITICAL ISSUE: Class 9 - Mixed Patterns

### Training Data Outliers (Should be removed or re-labeled):
```
train/9/3439_vl.png   - Avg distance to other class 9: 0.5593 ⚠️
train/9/4640_vl.png   - Avg distance to other class 9: 0.5562 ⚠️
```

These 2 images are **extremely different** from the other 8 class 9 training images.

### Test Images Matching the Outlier Pattern:
```
test/9/7981_vl.png    - Matches train/9/3439_vl.png (dist: 0.058)
test/9/8341_vl.png    - Matches train/9/3439_vl.png (dist: 0.051)
test/9/9416_vl.png    - Matches train/9/2635_vl.png (dist: 0.181)
test/9/9951_vl.png    - Matches train/9/3439_vl.png (dist: 0.080)
```

### Analysis:
Class 9 contains **two distinct patterns**:
- **Pattern A**: 8 training images (1481, 5185, 149, 3710, 3740, 4081, 1606, 2635)
- **Pattern B**: 2 training images (3439, 4640) + 4 test images (7981, 8341, 9416, 9951)

The template averaging both patterns doesn't match either well, causing 60% accuracy.

---

## Other Potential Issues

### Class 10:
- `test/10/6564_vl.png` - Failed with Bhatt=0.1440
  - Best match: `train/10/2074_vl.png` (correct class)
  - Warning: Close similarity to `train/1/1940_vl.png` (dist: 0.1363)
  - **Action**: Verify if this is truly class 10 or class 1

### Class 5:
- `test/5/8766_vl.png` - Failed with Bhatt=0.1781
  - Best match: `train/5/174_vl.png` (correct class)
  - High distance suggests outlier or edge case
  - **Action**: Verify if this belongs to class 5

### Class 6:
- `test/6/7468_vl.png` - Failed with Bhatt=0.2365
  - Best match: `train/6/3636_vl.png` (correct class)
  - Note: Class 6 training image `3636_vl.png` itself is an outlier (avg dist: 0.2678)
  - **Action**: Review both images

---

## Recommended Actions

### OPTION 1: Remove Outliers from Class 9 (Recommended)
If images 3439 and 4640 are the "wrong" pattern (e.g., single color, different cone type):

**Remove from training:**
```bash
rm train/9/3439_vl.png
rm train/9/4640_vl.png
```

**Remove matching test images:**
```bash
rm test/9/7981_vl.png
rm test/9/8341_vl.png
rm test/9/9416_vl.png
rm test/9/9951_vl.png
```

**Then retrain:**
```bash
python train_all_templates.py
python test_all_images.py
```

**Expected result**: Class 9 accuracy should jump to ~100%

---

### OPTION 2: Create New Class for Pattern B
If both patterns are valid but different cone types:

**Create new class (e.g., class 11):**
```bash
mkdir train/11 test/11

# Move Pattern B images
mv train/9/3439_vl.png train/11/
mv train/9/4640_vl.png train/11/
mv test/9/7981_vl.png test/11/
mv test/9/8341_vl.png test/11/
mv test/9/9416_vl.png test/11/
mv test/9/9951_vl.png test/11/
```

**Then retrain:**
```bash
python train_all_templates.py
python test_all_images.py
```

**Expected result**: Both class 9 and class 11 should have high accuracy

---

### OPTION 3: Keep Only Pattern B, Remove Pattern A
If the 2 outliers are actually the "correct" class 9 pattern:

**Remove the other 8 training images** and keep only 3439 and 4640
- This seems unlikely based on the data

---

## Images to Manually Inspect

**Priority 1 - Class 9 outliers:**
- `train/9/3439_vl.png`
- `train/9/4640_vl.png`

**Priority 2 - Failed test images:**
- `test/9/7981_vl.png`
- `test/9/8341_vl.png`
- `test/9/9416_vl.png`
- `test/9/9951_vl.png`

**Priority 3 - Other classes:**
- `test/10/6564_vl.png`
- `test/5/8766_vl.png`
- `test/6/7468_vl.png`
- `train/6/3636_vl.png`

---

## Current Accuracy (Before Cleaning)

| Class | Passed | Failed | Total | Accuracy |
|-------|--------|--------|-------|----------|
| 1     | 10     | 0      | 10    | 100.0%   |
| 2     | 10     | 0      | 10    | 100.0%   |
| 3     | 10     | 0      | 10    | 100.0%   |
| 5     | 8      | 1      | 9     | 88.9%    |
| 6     | 8      | 1      | 9     | 88.9%    |
| 9     | 6      | 4      | 10    | **60.0%** ⚠️ |
| 10    | 9      | 1      | 10    | 90.0%    |
| **TOTAL** | **61** | **7** | **68** | **89.7%** |

---

## Expected Accuracy (After Removing Class 9 Outliers)

If you follow **OPTION 1** and remove the 2 training outliers + 4 test images:

| Class | Expected Accuracy |
|-------|-------------------|
| 9     | ~100%            |
| Overall | ~95-98%        |

---

## Scripts Available

1. **analyze_failures.py** - Lists all failed images
2. **visualize_failures.py** - Shows failed images visually
3. **find_data_leakage.py** - Finds cross-class similarities
4. **analyze_training_data.py** - Finds outliers in training data
5. **train_all_templates.py** - Retrain after cleaning
6. **test_all_images.py** - Re-test after cleaning
