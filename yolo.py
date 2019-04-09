# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

from detector.detector import Detector
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

def get_yolo_path(filename):
	return os.path.sep.join([args["yolo"], filename])


def yolo():
	image = cv2.imread(args["image"])
	weightsPath = get_yolo_path("yolov3.weights")
	configPath = get_yolo_path("yolov3.cfg")
	labelsPath = get_yolo_path("coco.names")
	LABELS = open(labelsPath).read().strip().split("\n")
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),	dtype="uint8") 

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	# net = cv2.dnn.readNet(configPath, weightsPath)

	detector = Detector(net, image, args['confidence'], args['threshold'], LABELS, COLORS)
	img = detector.detect()

	cv2.imshow("Image", img)
	cv2.waitKey(0)

yolo()