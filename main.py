import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import find_radius, unrolled, find_object

img = cv2.imread(
    "/Users/nithinvadekkapat/work/cone_inspect/color_segment/data/9/1481_vl.png"
)
if img is None:
    print("Check Path")
else:
    # plt.imshow(img)
    # plt.show()
    pass

cropped_img, center, radius = find_radius.find_radius(img)
if cropped_img is not None:
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.show()
print(center, radius)
warpped_img = unrolled.unroll_cone_tip(cropped_img, center, radius)
print(warpped_img)
plt.imshow(cv2.cvtColor(warpped_img, cv2.COLOR_BGR2RGB))
plt.show()
result = find_object.find_object(warpped_img)
if result[0] is not None:
    (x_min, x_max), (y_min, y_max) = result
    warpped_img_crop = warpped_img[y_min : y_max + 1, x_min : x_max + 1]
    plt.imshow(cv2.cvtColor(warpped_img_crop, cv2.COLOR_BGR2RGB))
    plt.show()
