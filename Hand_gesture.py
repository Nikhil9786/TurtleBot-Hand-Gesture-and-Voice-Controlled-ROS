#!/usr/bin/env python
import cv2,time
import rospy
from geometry_msgs.msg import Twist
import math
import random
import numpy as np

# Open Camera
rospy.init_node('hand')
pub = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist, queue_size=1)
m = Twist()
rate = rospy.Rate(60)
capture = cv2.VideoCapture(0)

while(capture.isOpened()):
	
	# Capture frames from the camera
	ret, frame = capture.read()
	frame = cv2.flip(frame,1)
	
	# Get hand data from the rectangle sub window
	
	cv2.rectangle(frame, (80, 80), (250, 250), (0, 255, 0), 0)
	crop_image = frame[80:250, 80:250]

	# Apply Gaussian blur
	blur = cv2.GaussianBlur(crop_image, (5, 5), 0)

	# Change color-space from BGR -> HSV
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	# Create a binary image with where white will be skin colors and rest is black
	mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))

	# Kernel for morphological transformation
	kernel = np.ones((5, 5))

	# Apply morphological transformations to filter out the background noise
	
	dilation = cv2.dilate(mask2, kernel, iterations=1)
	erosion = cv2.erode(dilation, kernel, iterations=1)
	closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, np.ones((5,5)))

	# Apply Gaussian Blur and Threshold
	filtered = cv2.GaussianBlur(erosion, (5, 5), 0)
	ret, thresh = cv2.threshold(filtered,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Show threshold image
	cv2.imshow("Thresholded", thresh)
	# Find contours
	contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	try:
		# Find contour with maximum area
		contour = max(contours, key=lambda x: cv2.contourArea(x))

		# Create bounding rectangle around the contour
		x, y, w, h = cv2.boundingRect(contour)
		cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

		# Find convex hull
		hull = cv2.convexHull(contour)


		# Draw contour
		drawing = np.zeros(crop_image.shape,dtype =  np.uint8)
		cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
		cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

		# Find convexity defects
		hull = cv2.convexHull(contour, returnPoints=False)
		defects = cv2.convexityDefects(contour, hull)
		g = range(defects.shape[0])
		# Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
		# tips) for all defects
		count_defects = 0
		
		for i in g:
			s, e, f, d = defects[i, 0]
			start = tuple(contour[s][0])
			end = tuple(contour[e][0])
			far = tuple(contour[f][0])

			a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
			b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
			c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
			angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

			# if angle > 90 draw a circle at the far point
			if angle <= 90:
				count_defects += 1
				cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

			cv2.line(crop_image, start, end, [0, 255, 0], 2)

		# Print number of fingers
		if count_defects == 0:
			cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
			print('moving forward')
			m.linear.x = 0.5
		elif count_defects == 1:
			cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
			m.angular.z = 0.5
			print('LEFT')
		elif count_defects == 2:
			cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
			print('RIGHT')
			m.angular.z = -0.5
		elif count_defects == 3:
			cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
			m.linear.x = -0.5
		elif count_defects == 4:
			cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
		else:
			m.linear.x = 0
			m.angular.z = 0
			
		# Show required images
		pub.publish(m)
	except:
		pass
	
	cv2.imshow("Gesture", frame)
	all_image = np.hstack((drawing, crop_image))
	cv2.imshow('Contours', all_image)
	if cv2.waitKey(3) == ord('q'):
		break
	
capture.release()
