from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

cap = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


while(cap.isOpened()):
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	frame = imutils.resize(frame,width =min(400,frame.shape[1]))
	orig = frame.copy()
	(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
	for (x,y,w,h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
	rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
	pick = non_max_suppression(rects,probs = None, overlapThresh=0.65)
	for (xA,yA,xB,yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	cv2.imshow("Before",orig)
	cv2.imshow("After",frame)
	cv2.waitKey(0)	