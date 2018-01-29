import numpy as np
import cv2
import argparse
from imutils.paths import list_images
from frame_selector import FrameSelector

KEY_ESC = 27

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to images dataset...")
ap.add_argument("-a", "--annotations", required=True, default='annot.npy', help="path to save annotations...")
ap.add_argument("-i", "--images", required=True, default='images.npy', help="path to save images")
args = vars(ap.parse_args())

# annotations and image paths
annotations = []
im_paths = []

# loop through each image and collect annotations
for image_path in list_images(args["dataset"]):

    # load image and create a BoxSelector instance
    image = cv2.imread(image_path)
    bs = FrameSelector(image, "Image")
    cv2.imshow("Image", image)
    key_pressed = cv2.waitKey(0)

    if KEY_ESC == key_pressed:
        break

    # order the points suitable for the Object detector
    pt1, pt2 = bs.roiPts
    (x, y, xb, yb) = [pt1[0], pt1[1], pt2[0], pt2[1]]
    annotations.append([int(x), int(y), int(xb), int(yb)])
    im_paths.append(image_path)

# save annotations and image paths to disk
annotations = np.array(annotations)
im_paths = np.array(im_paths, dtype="unicode")
np.save(args["annotations"], annotations)
np.save(args["images"], im_paths)
