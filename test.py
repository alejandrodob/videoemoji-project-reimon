import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt

pattern = cv2.imread('pattern_circle.jpg', 0)
w, h = pattern.shape[::-1]
plot_output = "Purples"
folder = sys.argv[1]

images_in_dir = [f for f in os.listdir(folder) if f.endswith('.jpg') and "pattern" not in f]
print images_in_dir
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

meth = methods[5]


for img in images_in_dir:
    print img
    image = cv2.imread(folder+"/"+img, 0)
    img_copy = image.copy()
    img = img_copy.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,pattern,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = plot_output)
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = plot_output)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
