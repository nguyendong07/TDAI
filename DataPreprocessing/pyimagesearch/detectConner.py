import numpy as np
import cv2 as cv

img = cv.imread("C:/Users/ABC/Desktop/New folder/TDAI/Data/nhaqf.jpg")
new_img = cv.resize(img, (200, 200))
gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
conner = cv.goodFeaturesToTrack(gray, 30, 0.01, 0)
conner = np.int0(conner)

for i in conner:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, 255, -1)
cv.imshow('ot', img)
cv.waitKey(0)