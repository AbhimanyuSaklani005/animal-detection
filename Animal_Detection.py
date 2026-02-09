#!/usr/bin/env python
# coding: utf-8

# ****Real-Time Wildlife Detection and Alert System for Railway Tracks****

# *By: Abhimanyu Saklani(1NT23CS005), Nikhil Shah (1NT23CS148), Samarth Saxena (1NT23CS206)*

# **This project aims to prevent wildlife–train collisions by using a real-time object detection system that monitors railway tracks for the presence of animals. Leveraging a custom-trained YOLO model on Roboflow, the system processes live video feeds (e.g., from CCTV or webcams) to detect animals near or on the tracks. When an animal is identified, the system displays a visual alert and can optionally trigger further safety mechanisms such as alarm notifications. The model is optimized for various lighting conditions.**

# --------------------------------------------------------------

# *Install Required Libraries*

# In[1]:


get_ipython().system('pip install ultralytics opencv-python pillow ipywidgets matplotlib roboflow pyttsx3 --quiet')


# In[2]:


get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet')


# In[1]:


get_ipython().system('pip install roboflow --quiet')


# *Import Required Library*

# In[4]:


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import time
import pyttsx3
from roboflow import Roboflow
from IPython.display import clear_output, display


# In[ ]:





# In[8]:


rf = Roboflow(api_key="PolBl2dEq0yAqoPBBODy")
project = rf.workspace("realtime-wildlife-detection-and-alert-system-for-railway-tracks-animal-dataset").project("animal-detection-9dl1v")
version = project.version(1)
dataset = version.download("yolov8")


# In[3]:


print(dataset.location)


# *Load a base YOLOv8 model*

# In[10]:


from ultralytics import YOLO
model = YOLO('yolov8n.pt')


#  *Train your model using the 'C:\Users\abhi2\Animal-Detection--1\data.yaml' inside dataset folder with 50 Epoches*

# In[4]:


model.train(data=r'C:\Users\abhi2\Animal-Detection--1\data.yaml', epochs=50)


# 
# *Run the validation loop on the model and store the evaluation metrics*
# 

# In[7]:


metrics = model.val()


# In[27]:


model.export(format='torchscript')


# In[12]:


import shutil
shutil.make_archive('yolov8_model', 'zip', 'runs/detect/train212')


# In[14]:


from IPython.display import FileLink
FileLink(r'C:\\Users\\abhi2\\yolov8_model.zip')


# *Import YOLO from ultralytics and load the trained model weights from the specified path*

# In[7]:


from ultralytics import YOLO
model = YOLO(r'runs\detect\train21\weights\best.pt')


# *Animals Classes*

# In[60]:


animal_classes = ["Bear", "Buffalo", "Camel", "Cat", "Cow", "Deer", "Dog", "Donkey", "Elephant", "Leopard", "Lion", "Tiger"]


# *Loads an image, runs object detection using a pretrained YOLO model, and displays the original image alongside the detected objects with bounding boxes and labels*

# In[55]:


image_path = r"C:\Users\abhi2\Downloads\bu.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis('off')
plt.title("Input Image")
plt.show()
results = model(image_path)  
output_image = results[0].plot()
output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(output_image_rgb)
plt.axis('off')
plt.title("Detected Animals with Class Labels")
plt.show()


# In[ ]:


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

engine = pyttsx3.init()

animal_detected_time = {}
alerted_animals = set()

print("Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Histogram Equalization for better contrast
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized_gray = cv2.equalizeHist(gray_frame)
        enhanced_frame = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)

        # Run YOLOv8 detection on enhanced frame
        results = model(enhanced_frame)
        current_time = time.time()

        # Draw bounding boxes for detected animals
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label in animal_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # Yellow bounding box (BGR: 0,255,255)
                    cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(enhanced_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    # TTS alert after 2 seconds of continuous detection
                    if label not in animal_detected_time:
                        animal_detected_time[label] = current_time
                    elif (current_time - animal_detected_time[label]) > 2 and label not in alerted_animals:
                        print(f"🔊 Animal detected: {label}")
                        engine.say(f"Animal detected: {label}")
                        engine.runAndWait()
                        alerted_animals.add(label)

        # Show the frame with bounding boxes in separate window
        cv2.imshow('Animal Detection Webcam', enhanced_frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Quitting webcam...")
            break

except KeyboardInterrupt:
    print("🛑 Detection stopped by user")

finally:
    cap.release()
    engine.stop()
    cv2.destroyAllWindows()


# In[22]:


import cv2
from IPython.display import Video, display
def process_and_display_video_live(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {input_path}")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection on the frame
            results = model(frame)

            # Draw yellow bounding boxes and labels
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    if label in animal_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Convert BGR to RGB for displaying
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(frame_rgb)

            # Display frame in notebook output
            with io.BytesIO() as buf:
                pil_img.save(buf, format='jpeg')
                clear_output(wait=True)
                display(Image(data=buf.getvalue()))

            time.sleep(0.03)  # ~30 FPS display speed
        
    except KeyboardInterrupt:
        print("🛑 Video processing stopped by user")

    cap.release()
    print("✅ Video processing finished")


video_file_path = r"C:\Users\abhi2\Downloads\Lion charges scared tourist in car.mp4"  # Change to your local video path
process_and_save_video(video_file_path)


# In[26]:


# Read one frame from your video file
cap = cv2.VideoCapture(video_file_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to read frame")
else:
    # Run your model on this frame
    results = model(frame)

    # Print results to debug
    print("Detection results:", results)

    # Optionally draw detections on the frame
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            print(f"Detected: {label}")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the annotated frame
    from IPython.display import display
    import PIL.Image
    pil_img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    display(pil_img)


# In[68]:


import cv2

# Make sure you have your model and animal_classes defined before running this

def display_video_in_window(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {input_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # Run model detection on the frame
        results = model(frame)

        # Draw bounding boxes and labels in yellow
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label in animal_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frame in a new window
        cv2.imshow('YOLOv8 Detection', frame)

        # Press 'q' key to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage - change to your local video path
video_file_path = r"C:\Users\abhi2\Downloads\Elephant  short #video #shorts #funny #funnyanimal #elephant #animals #animalllover#youtubeshorts.mp4"
display_video_in_window(video_file_path)


# In[64]:


import cv2

# Make sure you have your model and animal_classes defined before running this

def display_video_in_window(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {input_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # Run model detection on the frame
        results = model(frame)

        # Draw bounding boxes and labels in yellow
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label in animal_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frame in a new window
        cv2.imshow('YOLOv8 Detection', frame)

        # Press 'q' key to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage - change to your local video path
video_file_path = r"C:\Users\abhi2\Downloads\Lion on Road # #shortsyoutube.mp4"
display_video_in_window(video_file_path)



# In[62]:


import cv2

# Make sure you have your model and animal_classes defined before running this

def display_video_in_window(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {input_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # Run model detection on the frame
        results = model(frame)

        # Draw bounding boxes and labels in yellow
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label in animal_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frame in a new window
        cv2.imshow('YOLOv8 Detection', frame)

        # Press 'q' key to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage - change to your local video path
video_file_path = r"C:\Users\abhi2\Downloads\Donkey on the road Animal Nature.mp4"
display_video_in_window(video_file_path)



# In[ ]:




