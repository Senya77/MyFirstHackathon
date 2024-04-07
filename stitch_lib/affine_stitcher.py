from stitching import AffineStitcher
import cv2
import sys

def stitch(file1, file2):
    settings = {"detector": "orb", "confidence_threshold": 0.0}
    stitcher = AffineStitcher(**settings)
    panorama = False
    try:
        panorama = stitcher.stitch([file1, file2])
    except:
        return False
    finally:
        return panorama
    
def main(path1, path2, name):

    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    stitched_image = stitch(image1, image2)
    res = cv2.imwrite(f'{name}.jpg', stitched_image)
    return res


if __name__ == '__main__':
    main('dataset\\1.jpg', 'dataset\\2.jpg', 'test')