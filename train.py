from detector import ObjectDetector
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--annotations", required=True, default='annot.npy', help="path to saved annotations...")
ap.add_argument("-i", "--images", required=True, default='images.npy', help="path to saved image paths...")
ap.add_argument("-d", "--detector", default='detector.svm',  help="path to save the trained detector...")
args = vars(ap.parse_args())

print "[INFO] loading annotations and images"
annotations = np.load(args["annotations"])
image_paths = np.load(args["images"])

detector = ObjectDetector()
print "[INFO] creating & saving object detector"

detector.fit(image_paths, annotations, visualize=True, save_path=args["detector"])
