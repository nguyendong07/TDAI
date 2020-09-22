#install dlib and fix some issue if don't sucees
import face_recognition
import cv2
img = face_recognition.load_image_file('input1.jpg')
face_locations = face_recognition.face_locations(img)
for face_location in face_locations:
  top_right, bottom_left, bottom_right, top_left, = face_location
  print(face_location)
  print(img.shape)

for i, bbox in enumerate(face_locations):
    bbox = list(map(int, bbox))
    top_right, bottom_left, bottom_right, top_left = bbox
    cv2.rectangle(img,(top_left, top_right),(bottom_left,bottom_right),(0,255,0),2)
cv2.imshow(img)
for i, bbox in enumerate(face_locations):
    bbox = list(map(int, bbox))
    top_right, bottom_left, bottom_right, top_left, = face_location
    crop_img = img[top_right:bottom_right, top_left:bottom_left]
cv2.imshow(crop_img)