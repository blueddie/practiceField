import numpy as np
import os
import tensorflow as tf
import cv2

# Load the label map
def load_label_map(label_map_path):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

# Load the model
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

# Perform object detection
def detect_objects(model, category_index, frame):
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections = model(input_tensor)

    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    height, width, _ = frame.shape

    for i in range(len(detection_boxes)):
        if detection_scores[i] > 0.5:
            box = detection_boxes[i] * np.array([height, width, height, width])
            (ymin, xmin, ymax, xmax) = box.astype(int)
            class_id = detection_classes[i]
            class_name = category_index[class_id]['name']
            if class_name == 'person':
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name}: {detection_scores[i]:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Main script
model_path = 'ssd_mobilenet_v2_coco_2018_03_29/saved_model'
# label_map_path = 'C:\Users\ed\Downloads/mscoco_label_map.pbtxt'
label_map_path = r'C:\Users\ed\Downloads\mscoco_label_map.pbtxt'
category_index = load_label_map(label_map_path)
detection_model = load_model(model_path)

# cap = cv2.VideoCapture("C:\Users\ed\Downloads\C_3_14_01_BU_DYA_07-19_14-34-34_c_DF6_M2.mp4")  # Replace with your video file
cap = cv2.VideoCapture = r'C:\Users\ed\Downloads\mscoco_label_map.pbtxt'
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_objects(detection_model, category_index, frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
