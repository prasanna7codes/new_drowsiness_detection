import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

# Define model path and image dimensions
MODEL_PATH = 'models/eye_state_model_new_data.h5'
IMG_WIDTH, IMG_HEIGHT = 24, 24

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print(f"Loaded trained model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run 'train_model.py' first.")
    exit()

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
alarm_on = False  # Flag to prevent repeated alarm triggers

while (True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    right_eye_detected = False
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (IMG_WIDTH, IMG_HEIGHT))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(IMG_WIDTH, IMG_HEIGHT, 1)
        r_eye = np.expand_dims(r_eye, axis=0)
        try:
            rpred_prob = model.predict(r_eye)[0]
            rpred_label = 1 if rpred_prob > 0.5 else 0
            rpred = [rpred_label]
            right_eye_detected = True
        except Exception as e:
            print(f"Error predicting right eye: {e}")
        break  # Process only the first detected right eye

    left_eye_detected = False
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (IMG_WIDTH, IMG_HEIGHT))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(IMG_WIDTH, IMG_HEIGHT, 1)
        l_eye = np.expand_dims(l_eye, axis=0)
        try:
            lpred_prob = model.predict(l_eye)[0]
            lpred_label = 1 if lpred_prob > 0.5 else 0
            lpred = [lpred_label]
            left_eye_detected = True
        except Exception as e:
            print(f"Error predicting left eye: {e}")
        break  # Process only the first detected left eye

    if right_eye_detected and left_eye_detected:
        if (rpred[0] == 0 and lpred[0] == 0):
            score += 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            alarm_on = False  # Reset alarm flag when eyes are open
    elif right_eye_detected and not left_eye_detected:
        if (rpred[0] == 0):
            score += 1
            cv2.putText(frame, "Right Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Right Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            alarm_on = False
    elif not right_eye_detected and left_eye_detected:
        if (lpred[0] == 0):
            score += 1
            cv2.putText(frame, "Left Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Left Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            alarm_on = False
    else:
        cv2.putText(frame, "No Eyes Detected", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        score = max(0, score - 2)
        alarm_on = False

    if score < 0:
        score = 0
        alarm_on = False
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score > 15 and not alarm_on:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
            alarm_on = True  # Set the flag to indicate alarm is playing
        except Exception as e:
            print(f"Error playing sound: {e}")
            pass
        thicc = min(thicc + 2, 16)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    elif score <= 15:
        thicc = 2
        alarm_on = False # Reset alarm flag if score drops below threshold

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()