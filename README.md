# MPVTracker
Real-Time Person, Car &amp; Truck Tracker using YOLOv5 A Python-based real-time object detection and tracking system using YOLOv5. Detects and tracks people, cars, and trucks from webcam or video input, with unique IDs, confidence scores, and automatic removal of lost targets.

![image](https://github.com/user-attachments/assets/8c400bcc-3fb5-4db4-ac8f-2421bfe523b6)


### Developer notes:
For live tracking would the optimal hardware be:
- GPU: NVIDIA RTX 3080/3090 or RTX 40-series
- CPU: Intel i9 / AMD Ryzen 9
- RAM: 32 GB+
- Camera Input: IP cameras or high-def USB/HDMI feeds (via capture cards)
- Cooling: Dedicated GPU & CPU cooling if running long sessions

I tested it on a intel core5 8th gen, which was very slow. the pictures and videos are downloaded video.

### Requirements
yolov5(s/x) dependencies
matplotlib
Pillow
PyYAML
requests
scipy

torch>=1.7
opencv-python
numpy

Usually installed with YOLOV5 
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or choose GPU version
pip install opencv-python numpy matplotlib Pillow PyYAML requests scipy

```

### Usage
```
python3 MPTracker2.py <videofile.mp4>
```


![image](https://github.com/user-attachments/assets/1f561811-b894-42fc-a83e-403daa100016)




