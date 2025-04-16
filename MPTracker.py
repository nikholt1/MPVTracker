import cv2
import numpy as np
import sys
import argparse
import os

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description="Multi-person and vehicle tracker")
parser.add_argument("-w", "--video", type=str, help="Path to video file (default: webcam)", default=None)
parser.add_argument("-f", "--frequency", type=int, help="Detection frequency in frames", default=5)
args = parser.parse_args()

video_path = args.video
detect_every = args.frequency

# --- Video Source ---
if video_path:
    cap = cv2.VideoCapture(video_path)
    print(f"🎥 Using video file: {video_path}")
else:
    cap = cv2.VideoCapture(0)
    print("🎥 Using live camera")

if not cap.isOpened():
    print("❌ Video source can't be opened")
    sys.exit(1)

# --- HOG detector setup (People detection) ---
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --- YOLO setup (Military vehicle detection) ---
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
vehicle_classes = ['car', 'truck', 'bus', 'motorbike']

# --- Tracker setup ---
trackers = []
frame_count = 0

# --- Unique person tracking ---
seen_boxes = []
total_people_count = 0
iou_threshold = 0.3

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Use a tracker that's generally available
# --- Tracker selection with fallback ---
if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
    tracker_type = cv2.legacy.TrackerCSRT_create
elif hasattr(cv2, 'TrackerCSRT_create'):
    tracker_type = cv2.TrackerCSRT_create
elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
    tracker_type = cv2.legacy.TrackerKCF_create
elif hasattr(cv2, 'TrackerKCF_create'):
    tracker_type = cv2.TrackerKCF_create
elif hasattr(cv2, 'TrackerMIL_create'):
    tracker_type = cv2.TrackerMIL_create
else:
    print("❌ No compatible tracker found in your OpenCV installation.")
    sys.exit(1)
 # works with `opencv-contrib-python`

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ End of video or cannot read frame.")
        break

    frame_count += 1

    if frame_count % detect_every == 0:
        trackers = []

        # --- People detection using HOG ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8))

        for (x, y, w, h) in boxes:
            new_box = (x, y, w, h)
            matched = any(compute_iou(new_box, seen_box) > iou_threshold for seen_box in seen_boxes)

            if not matched:
                seen_boxes.append(new_box)
                total_people_count += 1
                print(f"New person detected! Total unique people: {total_people_count}")

            tracker = tracker_type()
            trackers.append((tracker, new_box))
            trackers[-1][0].init(frame, new_box)

        # --- Military vehicle detection using YOLO ---
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in vehicle_classes:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    x = center_x - w // 2
                    y = center_y - h // 2

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"Vehicle: {confidence:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Update trackers
    for tracker, _ in trackers:
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {trackers.index((tracker, _)) + 1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            print("❌ Tracker failed to update.")

    cv2.imshow("Multi-Person & Vehicle Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



