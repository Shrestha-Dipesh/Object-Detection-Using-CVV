import cv2 #pip install opencv-python

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(config_path, weights_path)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

classLabels = []
file_name = 'coco.names'
with open(file_name, 'rt') as file:
    classLabels = file.read().rstrip('\n').split('\n')

while True:
    success, image = cap.read()
    classIds, confidences, bbox = model.detect(image, confThreshold = 0.5)

    for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), bbox):
        cv2.rectangle(image, box, color = (255, 0, 0), thickness = 2)
        cv2.putText(image, classLabels[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(round(confidence * 100)) + '%', (box[0] + 250, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    if (key == ord('q')):
        break