import cv2
import numpy as np
import csv
from datetime import datetime
from tflite_runtime.interpreter import Interpreter
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
interpreter = Interpreter(model_path='/home/pi/Desktop/Real Time Face Emotion Recgonition/my_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

csv_path = '/home/pi/Desktop/Real Time Face Emotion Recgonition/emotion_log.csv'

csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
max_faces = 5  
header = ['Timestamp'] + [f'Face_{i}' for i in range(max_faces)]
csv_writer.writerow(header)
csv_file.close()

cap = cv2.VideoCapture("http://YOUR IP/?action=stream")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    emotions_row = [''] * max_faces 
    face_id = 0

    for (x, y, w, h) in faces:
        if face_id >= max_faces:
            break 

        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            roi_resized = cv2.resize(roi, (48, 48)).reshape(1, 48, 48, 1).astype('float32') / 255.0
            interpreter.set_tensor(input_details[0]['index'], roi_resized)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            emotion_idx = np.argmax(output_data)
            emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][emotion_idx]

            emotions_row[face_id] = emotion_label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Face {face_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, emotion_label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            face_id += 1

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [timestamp] + emotions_row
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    cv2.imshow("Real-Time Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

os.system(f"xdg-open '{csv_path}'")
