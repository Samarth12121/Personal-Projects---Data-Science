import streamlit as st
from PIL import Image
import cv2
from app import initialize, capture_frame, preprocess_frame, detect_faces, detect_eyes, preprocess_eye, predict_drowsiness, display_results, release_resources, stop_alarm

def run_drowsiness_detection():
    global alarm_on
    alarm_on = False
    labels = ['Closed', 'Open']
    model, face_cascade, eye_cascade, cap = initialize()

    st.title("Real-Time Drowsiness Detection")
    run = st.checkbox('Run Drowsiness Detection')
    FRAME_WINDOW = st.image([])

    while run:
        frame = capture_frame(cap)
        if frame is None:
            st.write("Failed to capture video")
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
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        FRAME_WINDOW.image(img)

    release_resources(cap)
    stop_alarm()

if __name__ == "__main__":
    run_drowsiness_detection()
