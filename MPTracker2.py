import cv2
import torch
import sys
import warnings

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', trust_repo=True)

# COCO class labels
CLASS_NAMES = model.names  # e.g., 0: 'person', 2: 'car', 7: 'truck', etc.

# Get video source
if len(sys.argv) > 1:
    source = sys.argv[1]  # Video file
else:
    source = 0  # Webcam

# Open video source
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"❌ Error: Unable to open video source: {source}")
    sys.exit()

print(f"📹 Tracking from: {'Webcam' if source == 0 else source}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Done or no frames left.")
        break

    # Run detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    person_count = 0
    car_count = 0
    truck_count = 0

    # Draw and log detections
    for *box, conf, cls in detections:
        label = CLASS_NAMES[int(cls)]
        conf = float(conf)

        if label in ['person', 'car', 'truck']:
            print(f"[INFO] Detected {label} with confidence: {conf:.2f}")

            if label == 'person':
                person_count += 1
            elif label == 'car':
                car_count += 1
            elif label == 'truck':
                truck_count += 1

            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if label == 'person' else (255, 255, 0) if label == 'car' else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Show total counts
    count_text = f'People: {person_count} | Cars: {car_count} | Trucks: {truck_count}'
    cv2.putText(frame, count_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("YOLOv5 Tracker", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



