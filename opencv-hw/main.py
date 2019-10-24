# -*- coding: utf-8 -*-
import cv2
import numpy as np

"""## Draw a triangle and write Homework 2 in the middle of the triangle."""

width = 500
height = 500

img = np.full((height, width, 3), 255, np.uint8)

p1 = (int(height / 2), 0)
p2 = (0, width)
p3 = (height, width)
cv2.line(img, p1, p2, (0, 0, 0), 3)
cv2.line(img, p2, p3, (0, 0, 0), 3)
cv2.line(img, p1, p3, (0, 0, 0), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
text = "Homework 2"
text_size = cv2.getTextSize(text, font, 1, 2)[0]
textX = int((img.shape[1] - text_size[0]) / 2)
textY = int((img.shape[0] + text_size[1]) / 2)

cv2.putText(img, text, (textX, textY), font, 1, (0, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)

"""## Count how many coins are in the picture."""

img = cv2.imread('img/coins.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.medianBlur(gray, 5)
detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=60, minRadius=1,
                                    maxRadius=150)

if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
font = cv2.FONT_HERSHEY_SIMPLEX
text = str(len(detected_circles[0, :])) + " coins"
text_size = cv2.getTextSize(text, font, 1, 2)[0]
textX = int((img.shape[1] - text_size[0]) / 2)
textY = int((img.shape[0] + text_size[1]) / 2)

cv2.putText(img, text, (textX, textY), font, 1, (0, 0, 255), 4)
cv2.imshow('img', img)
cv2.waitKey(0)

"""## Road Lane Lines Detection"""

img = cv2.imread('img/road.jpg', cv2.IMREAD_COLOR)  # road.png is the filename
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 250, minLineLength=250, maxLineGap=250)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
cv2.imshow('img', img)
cv2.waitKey(0)

"""## Face Detection follows slides."""

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('img/cherprang.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
