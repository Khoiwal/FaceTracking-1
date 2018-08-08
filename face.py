from imutils.video import VideoStream
from pyimagesearch.centroidtracker import CentroidTracker
import unicornhathd
import imutils
import numpy as np
import cv2
import sys
import os
import math

import time

model = "res10_300x300_ssd_iter_140000.caffemodel"
prototxt = "deploy.prototxt.txt"
ct = CentroidTracker()
(H, W) = (None, None)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

try:
    print("[INFO] starting face tracking")
    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (W, H),
            (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            rects.append(box.astype("int"))

        objects = ct.update(rects)

        for(objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            print(text)
            print(centroid)

    	key = cv2.waitKey(1) & 0xFF

    	# if the `q` key was pressed, break from the loop
    	if key == ord("q"):
    		break

    vs.stop()


except KeyboardInterrupt:
    unicornhathd.off()
    print("Face Tracking Ending")
