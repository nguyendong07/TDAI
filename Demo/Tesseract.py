import pytesseract
import cv2
import numpy as np
img = cv2.imread('C:/Users/ABC/Desktop/New folder/TDAI/Data/vanban.jpg')

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image, 5)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iteration=1)

def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iteration=1)

def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
       angle = -(90 + angle)
    else:
       angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

gray = get_grayscale(img)
thres = thresholding(gray)
open = opening(thres)
canny = canny(open)
# cv2.imshow('gray',gray)
# cv2.waitKey(0)
# cv2.imshow('thres',thres)
# cv2.waitKey(1)
# cv2.imshow('open',open)
# cv2.waitKey(2)
# cv2.imshow('canny',canny)
# cv2.waitKey(3)

custom_config = r'--oem 3 --psm 6 lang = vi'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
print(pytesseract.image_to_string(gray, config=custom_config))
