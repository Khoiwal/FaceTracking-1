import cv2
import sys
import os
import math
from threading import Thread


class UIUpdate(Thread):
    def __init__(self, faceTracking):
        Thread.__init__(self)
        self.daemon = True

    def run(self):
        while True:
            faces = faceTracking.getFaces()
            print(faces)


class FaceTrack(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        cascPath = "./face.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.video_capture = cv2.VideoCapture(0)
        self.width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.faces = []

    def getFaces(self):
        return self.faces;

    def run(self):

        while True:

            ret, frame = self.video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            self.faces = [];

            for (x, y, w, h) in faces:
                face_center_w = x + (w/2)
                face_center_h = y + (h/4)
                percentage_x = 100 - ((face_center_w/self.width) * 100)
                percentage_y = (face_center_h / self.width) * 100
                self.faces.append([percentage_x, percentage_y])

        self.video_capture.release()

try:
    print("Starting face tracking")
    faceTracking = FaceTrack()
    faceTracking.start()

    ui = UIUpdate(faceTracking)
    ui.start()
    while True:
        pass

except KeyboardInterrupt:
    print("Face Tracking Ending")
