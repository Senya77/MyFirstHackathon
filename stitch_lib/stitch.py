from stitching import Stitcher
from stitching import AffineStitcher
import cv2
import matplotlib.pyplot as plt

settings = {"detector": "orb", "confidence_threshold": 0.0}
stitcher = AffineStitcher(**settings)

file1 = cv2.imread('hack_stitch_dataset\\1\\1.jpeg')
file2 = cv2.imread('hack_stitch_dataset\\1\\2.jpeg')

panorama = None
try:
    panorama = stitcher.stitch([cv2.imread(file1), cv2.imread(file2)])
except:
    pass

if panorama is not None:
    print(1)
    cv2.imwrite('nap1.jpg', panorama)