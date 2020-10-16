import cv2
from detecto import core, utils, visualize
import torch
import numpy as np
from scipy.spatial.distance import euclidean


model = torch.load('cropv2')
fname = 'Test.jpg'
image = utils.read_image(fname)
labels, boxes, scores = model.predict(image)
# print(labels)
print(boxes)
# for i, bbox in enumerate(boxes):
#     bbox = list(map(int, bbox))
#     x_min, y_min, x_max, y_max = bbox
#     cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,255,0),2)
#     cv2.putText(image, labels[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
# cv2.imshow('img',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# merge overlap boxes
def non_max_suppression_fast(boxes, labels, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    #
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")

    return final_boxes, final_labels

#return final result
final_boxes, final_labels = non_max_suppression_fast(boxes.numpy(), labels, 0.15)
for i, bbox in enumerate(final_boxes):
    bbox = list(map(int, bbox))
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,255,0),2)
    cv2.putText(image, labels[i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#find centerpoint
def get_center_point(box):
    xmin, ymin, xmax, ymax = box
    return (xmin + xmax) // 2, (ymin + ymax) // 2
final_points = list(map(get_center_point, final_boxes))
label_boxes = dict(zip(final_labels, final_points))
print(final_labels)
print(label_boxes)

# def rotate(coor):
#   br,bl,tr,tl = [coor['bottom_right'],coor['bottom_left'],coor['top_right'],coor['top_left']]
#   print(br[0])
#   dis1=euclidean(coor['bottom_right'],coor['emblem']) # kc euclid tu bottom right den emblem
#   dis2=euclidean(coor['bottom_left'],coor['emblem']) # kc euclid tu bottom left den emblem
#   dis3=euclidean(coor['top_right'],coor['emblem'])
#   dis4=euclidean(coor['top_left'],coor['emblem'])
#   a=[dis1,dis2,dis3,dis4]
#   index= 0
#   minCoor = a[0]
#   i = 0
#   # tim khoang cach ngan nhat tu quoc huy(emblem) => cac toa do goc/ output = index
#   while(i<len(a)):
#     if(minCoor > a[i]):
#       minCoor = a[i]
#       index = i
#     i= i+1
#   # index = 0,1,2,3: quoc huy o goc: br,bl,tr,tl
#   print(index)
#   if (index==0 ):
#     return np.float32([br,bl,tl,tr])
#   elif (index==1):
#     return np.float32([bl,tl,tr,br])
#   elif (index==2):
#     return np.float32([tr,br,bl,tl])
#   else: return np.float32([tl,tr,br,bl])
# print(rotate(label_boxes))

# #crop image
# def perspective_transoform(image, source_points):
#     print(source_points)
#     dest_points = np.float32([[0,0], [500,0], [500,300], [0,300]])
#     M = cv2.getPerspectiveTransform(source_points, dest_points)
#     dst = cv2.warpPerspective(image, M, (500, 300))
#     return dst
# # return image after crop
# source_points = rotate(label_boxes)
# # Transform
# crop = perspective_transoform(image, source_points)
# cv2.imshow('ot',crop)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# crop text area
# def CropTextArea(image,coordinate,H,W):
#   return image[coordinate[0]:coordinate[0]+W,coordinate[1]:coordinate[1]+H]
#
# print(crop.shape)
# caculate frames of text
# iHeight = crop.shape[1]
# iWidth = crop.shape[0]
   ## name frame
# def CaculateNameFrame(image,h,w):
#   coordinate = (np.int32((2.2/7.5)*w),np.int32((3.7/12.5)*h))
#   width = np.int32(1.6/7.5*w )
#   height= np.int32( 9/12.5 *h )
#   return (coordinate, height, width)
# def CaculateIdFrame(image,h,w):
#   coordinate = (np.int32((1.4/7.5)*w),np.int32((5/12.5)*h))
#   width = np.int32(1.5/7.5*w )
#   height= np.int32( 6.5/12.5 *h )
#   return (coordinate, height, width)
# def CaculateDateFrame(image,h,w):
#   coordinate = (np.int32((3.8/7.5)*w),np.int32((3.7/12.5)*h))
#   width = np.int32(1.1/7.5*w )
#   height= np.int32( 9/12.5 *h )
#   return (coordinate, height, width)
# def CaculateAddress1Frame(image,h,w):
#   coordinate = (np.int32((4.7/7.5)*w),np.int32((3.7/12.5)*h))
#   width = np.int32(1.5/7.5*w )
#   height= np.int32( 9/12.5 *h )
#   return (coordinate, height, width)
# def CaculateAddress2Frame(image,h,w):
#   coordinate = (np.int32((5.8/7.5)*w),np.int32((3.5/12.5)*h))
#   width = np.int32(1.7/7.5*w )
#   height= np.int32( 9/12.5 *h )
#   return (coordinate, height, width)
#
# (coordinate, height, width) = CaculateIdFrame(crop,iHeight,iWidth)
# (coordinate3, height3, width3) = CaculateNameFrame(crop,iHeight,iWidth)
# (coordinate4, height4, width4) = CaculateDateFrame(crop,iHeight,iWidth)
# (coordinate5, height5, width5) = CaculateAddress1Frame(crop,iHeight,iWidth)
# (coordinate6, height6, width6) = CaculateAddress2Frame(crop,iHeight,iWidth)
#
# image1 = CropTextArea(crop,coordinate, height, width)
# image3 = CropTextArea(crop,coordinate3, height3, width3)
# image4 = CropTextArea(crop,coordinate4, height4, width4)
# image5 = CropTextArea(crop,coordinate5, height5, width5)
# image6 = CropTextArea(crop,coordinate6, height6, width6)
# cv2.imshow(image1)
# cv2.imshow(image3)
# cv2.imshow(image4)
# cv2.imshow(image5)
# cv2.imshow(image6)

