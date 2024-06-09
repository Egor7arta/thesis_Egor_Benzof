import streamlit as st
from main import run_license_plate_recognition
import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

# Constants
pattern = r'\b(?:[A-Z]{1}[A-Z]{2}\d{2,3}[A-Z]{1,2}\d{4}|[A-Z]{2}\d{2,3}[A-Z]{1,2}\d{4})\b'
frame_interval = 0.5

# Page configuration
st.set_page_config(
    page_title="Система распознавания гос. регистрационных знаков",
    page_icon="🚗",
    layout="wide",
)

# Display project information
st.title("Модель интеграции сторонних автомобильных номеров в систему распознавания государственных регистрационных знаков")
st.subheader("Университет: СПБГУТ им. Бонч-Бруевича")
st.write("**Код профессии:** 09.03.04")
st.write("**Имя преподавателя:** Бирюков Михаил Александрович")
st.write("**Группа:** ИКПИ-06")
st.write("**ФИО:** Артамонов Егор Андреевич")

# Main application
def app():
    st.header("Система распознавания государственных регистрационных знаков")
    st.subheader("Powered by YOLOv5")
    st.write("Добро пожаловать!")

    with st.form("my_uploader"):
        uploaded_file = st.file_uploader(
            "Upload file", type=["png", "jpg", "jpeg", "mp4"], accept_multiple_files=False
        )
        submit = st.form_submit_button(label="Выгрузить")

        if uploaded_file is not None:
            # Create directory if it doesn't exist
            if not os.path.exists("temp"):
                os.makedirs("temp")

            # Save uploaded file
            save_path = os.path.join("temp", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Check if the uploaded file is a video
            if uploaded_file.type.startswith('video'):
                # Process video here (e.g., display or analyze frames)
                st.video(uploaded_file)

                if submit and uploaded_file is not None:
                    # Add spinner
                    with st.spinner(text="Обнаружение осударственных регистрационных знаков ..."):
                        if not os.path.exists("frame_dir"):
                            os.makedirs("frame_dir")

                        cap = cv2.VideoCapture(save_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_interval_frames = int(fps * frame_interval)

                        captured_plates = []
                        captured_time = []
                        frame_number = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            if frame_number % frame_interval_frames == 0:
                                # Append the frame to the list
                                frame_filename = os.path.join("frame_dir", f"frame_{frame_number + 1}.png")
                                cv2.imwrite(frame_filename, frame)

                                recognizer = run_license_plate_recognition(frame_filename)
                                text = recognizer.recognize_text()
                                if text:
                                    matches = re.search(pattern, text)
                                    if matches:
                                        trimmed_string = matches.group(0)
                                        trimmed_string = re.sub(r'^E(?=K)', '', trimmed_string)
                                    else:
                                        trimmed_string = None
                                    if trimmed_string:
                                        st.write(f"Обнаружен гос. номер {frame_number}: {trimmed_string}")
                                        captured_plates.append(trimmed_string)
                                        captured_time.append(frame_number)

                            frame_number += 1

                        # Release the video capture object and delete the temporary directory
                        cap.release()
                        st.warning("Frames extraction complete.")
                        df = pd.DataFrame({'Обнаруженный гос номер:': captured_plates, 'Time Stamps': captured_time})
                        df['Serial Number'] = range(1, len(df) + 1)

                        st.write(df)

            else:
                # Process image here (e.g., display or analyze image)
                st.image(save_path)
                if submit and uploaded_file is not None:
                    # Add spinner
                    with st.spinner(text="Обнаружение гос номера ..."):
                        recognizer = run_license_plate_recognition(save_path)
                        text = recognizer.recognize_text()
                        if text:
                            st.write(f"Обнаруженный гос. номер: {text}")

if __name__ == "__main__":
    app()
