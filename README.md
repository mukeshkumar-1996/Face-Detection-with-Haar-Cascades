# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program 

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# PART 1: ROI SEGMENTATION
# =========================

image = cv2.imread('vijay.jpg')

if image is None:
    print("Error: vijay.jpg not found")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# ROI
roi = image[100:420, 200:550]

mask = np.zeros_like(image)
mask[100:420, 200:550] = roi

segmented = cv2.bitwise_and(image, mask)

plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("Segmented ROI")
plt.axis('off')
plt.show()


# =========================
# PART 2: EDGE DETECTION
# =========================

image = cv2.imread('vijay.jpg')

if image is None:
    print("Error: vijay.jpg not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = image.copy()
          
for c in contours:
    if cv2.contourArea(c) > 50:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Contour Detection")
plt.axis('off')
plt.show()


# =========================
# PART 3: OBJECT DETECTION (SAFE VERSION)
# =========================

config_file = 'deploy.prototxt'
weights_file = 'mobilenet_iter_73000.caffemodel'

# If model files NOT found → skip safely
if not os.path.exists(config_file) or not os.path.exists(weights_file):
    print("⚠️ Model files not found → Skipping Object Detection part")
else:
    net = cv2.dnn.readNetFromCaffe(config_file, weights_file)

    class_labels = {
        0:'background',1:'aeroplane',2:'bicycle',3:'bird',4:'boat',
        5:'bottle',6:'bus',7:'car',8:'cat',9:'chair',10:'cow',
        11:'diningtable',12:'dog',13:'horse',14:'motorbike',
        15:'person',16:'pottedplant',17:'sheep',18:'sofa',
        19:'train',20:'tvmonitor'
    }

    image = cv2.imread('vijay.jpg')

    if image is None:
        print("Error: vijay.jpg not found")
        exit()

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = class_labels.get(idx, "Unknown")

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Object Detection (MobileNet-SSD)")
    plt.axis('off')
    plt.show()
```

## Output
<img width="519" height="521" alt="Screenshot 2026-09-16 172109" src="https://github.com/user-attachments/assets/e89d3888-05e8-4cad-82b7-b034ff86117f" />

<img width="507" height="514" alt="Screenshot 2026-09-16 172133" src="https://github.com/user-attachments/assets/594d7585-8fe3-4a29-a653-9249f19f4d2b" />

<img width="495" height="518" alt="Screenshot 2026-09-16 172145" src="https://github.com/user-attachments/assets/0e74749d-1d27-4af6-9e78-778a957f3b6c" />

<img width="501" height="513" alt="Screenshot 2026-09-16 172152" src="https://github.com/user-attachments/assets/86fbe77b-9372-4d66-a35e-1d19a0c1eae9" />


## Result 

Thus to write a Python program using OpenCV to perform the following image manipulations was verified successfully.


