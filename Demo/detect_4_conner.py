import cv2
from detecto import core, utils, visualize
import numpy as np

dataset = core.Dataset('C:/Users/ABC/Desktop/New folder/TDAI/Data/sample') #load data
print(len(dataset))
model = core.Model(['top_left', 'top_right', 'bottom_left', 'bottom_right']) #download model

losses = model.fit(dataset, epochs=30, verbose=True, learning_rate=0.001) #set parameter formodel and fit

model.save('id_card_4_conner.pth')

frame = ''
image = utils.read_image(frame)
labels, boxes, score = model.predict(image)
print(labels)
print(boxes)

##draw boxes
for i, box in enumerate(boxes):
    box = list(map(int, box))
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, labels[i], (x_min, y_min), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))

#nonmax suppression
def non_max_supperession_fast(boxes, labels, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)


    while len(idxs) > 0:
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx1 - xx2 + 1)
        h = np.maximum(0, yy1 - yy2 + 1)

        overlap = (w * h)/ area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        final_label = [labels[idxs] for idx in pick]
        final_boxes = boxes[pick].astype("int")

        return final_boxes, final_label

final_boxes, final_label = non_max_supperession_fast(boxes.numpy(), labels, 0.15)

def get_center_point(box):
    xmin, ymin, xmax, ymax = box
    return (xmin + xmax) // 2, (ymin + ymax)//2

final_point = list(map(get_center_point, final_boxes))

label_boxes = dict(zip(final_label, final_point))

#tranform sang toa do dich

def perspective_transform(image, source_points):
    dest_point = np.float32([[0,0], [500,500], [500,300], [00,300]])
    M = cv2.getPerspectiveTransform(source_points, dest_point)
    dst = cv2.warpPerspective(image, M, (500, 300))
    return dst

source_point = np.float32([label_boxes['top_left'], label_boxes['top_right'], label_boxes['bottom_left'], label_boxes['bottom_right']])

crop = perspective_transform(image, source_point)
