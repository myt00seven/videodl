import imageio
images = []

import cv2
import numpy as np
import matplotlib.pyplot as plt



ITER_FIRST = 1
# ITER_STEP = 1
ITER_LAST = 100

STR_PATH = "data/"
STR_FILE = "reconst_images_"
STR_SUFFIX = ".png"

filenames=[]

for i in range(ITER_FIRST,ITER_LAST+1):
	# filenames.append(STR_PATH+STR_FILE+str(i)+STR_SUFFIX)
    filename = STR_PATH+STR_FILE+str(i)+STR_SUFFIX
# font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 25)
# for filename in filenames:
    im = imageio.imread(filename)
    # cv2.putText(img=im, text=str(i), org=(180,385),fontFace=2, fontScale=1, color=(255,0,0), thickness=2)
    cv2.putText(img=im, text=str(i), org=(40,80),fontFace=2, fontScale=1, color=(255,0,0), thickness=2)
    images.append(im)

imageio.mimsave('./CIFAR10.gif', images)