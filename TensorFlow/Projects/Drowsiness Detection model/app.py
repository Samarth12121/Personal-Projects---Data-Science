import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import threading

alarm_on = False
def initialize():
    model = load_model('Driver_Drowsiness_Detection.keras')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)
    pygame.mixer.init()
    return model, face_cascade, eye_cascade, cap


def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def detect_faces(face_cascade, gray_frame):
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    return faces


def detect_eyes(eye_cascade, gray_frame, face_coords):
    (x, y, w, h) = face_coords
    roi_gray = gray_frame[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    return [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]


def preprocess_eye(eye_img):
    eye_img = cv2.resize(eye_img, (32, 32))
    eye_img = np.expand_dims(eye_img, axis=0)
    return eye_img


def predict_drowsiness(model, eye_img):
    prediction = model.predict(eye_img)
    return prediction


def play_alarm():
    pygame.mixer.music.load('alarm_sound.wav')
    pygame.mixer.music.play(-1)


def stop_alarm():
    pygame.mixer.music.stop()


def display_results(frame, faces, eyes, predictions, labels):
    global alarm_on
    drowsy_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (ex, ey, ew, eh), pred in zip(eyes, predictions):
            label = labels[np.argmax(pred)]
            color = (0, 255, 0) if label == 'Open' else (0, 0, 255)
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(frame, label, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if label == 'Closed':
                cv2.putText(frame, 'Drowsiness Alert!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                drowsy_detected = True

    if drowsy_detected and not alarm_on:
        alarm_on = True
        threading.Thread(target=play_alarm).start()
    elif not drowsy_detected and alarm_on:
        alarm_on = False
        stop_alarm()

    return frame


def release_resources(cap):
    cap.release()
    cv2.destroyAllWindows()


def main():
    global alarm_on
    alarm_on = False
    labels = ['Closed', 'Open']
    model, face_cascade, eye_cascade, cap = initialize()

    while True:
        frame = capture_frame(cap)
        if frame is None:
            break

        gray_frame = preprocess_frame(frame)
        faces = detect_faces(face_cascade, gray_frame)

        all_eyes = []
        predictions = []
        for face in faces:
            eyes = detect_eyes(eye_cascade, gray_frame, face)
            all_eyes.extend(eyes)
            for (ex, ey, ew, eh) in eyes:
                eye_img = preprocess_eye(frame[ey:ey + eh, ex:ex + ew])
                pred = predict_drowsiness(model, eye_img)
                predictions.append(pred)

        frame = display_results(frame, faces, all_eyes, predictions, labels)
        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_resources(cap)
    stop_alarm()

if __name__ == "__main__":
    main()
