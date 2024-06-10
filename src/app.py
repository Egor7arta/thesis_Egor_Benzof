import streamlit as st
import sqlite3
from main import run_license_plate_recognition
import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Database setup
conn = sqlite3.connect('license_plates.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS plates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT UNIQUE,
        full_name TEXT,
        address TEXT
    )
''')
conn.commit()

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

# Initialize session state for form inputs
if 'plate_number' not in st.session_state:
    st.session_state['plate_number'] = ''
if 'full_name' not in st.session_state:
    st.session_state['full_name'] = ''
if 'address' not in st.session_state:
    st.session_state['address'] = ''
if 'save_success' not in st.session_state:
    st.session_state['save_success'] = False

# Main application
def app():
    st.header("Система распознавания государственных регистрационных знаков")
    st.subheader("Powered by YOLOv5")
    st.write("Добро пожаловать!")

    uploaded_file = st.file_uploader("Загрузить файл", type=["png", "jpg", "jpeg", "mp4"], accept_multiple_files=False)
    if uploaded_file is not None:
        # Create directory if it doesn't exist
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Save uploaded file
        save_path = os.path.join("temp", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process image here (e.g., display or analyze image)
        st.image(save_path)
        if st.button("Обработать файл"):
            # Add spinner
            with st.spinner(text="Обнаружение гос номера ..."):
                recognizer = run_license_plate_recognition(save_path)
                text = recognizer.recognize_text()
                st.write(f"Распознанный текст: {text}")  # Debug output
                if text:
                    trimmed_string = text.strip()
                    st.session_state['plate_number'] = trimmed_string
                    st.write(f"Обнаруженный гос. номер: {trimmed_string}")

                    # Check database for existing plate number
                    c.execute("SELECT full_name, address FROM plates WHERE plate_number=?", (trimmed_string,))
                    result = c.fetchone()
                    if result:
                        st.write(f"Информация о владельце: ФИО - {result[0]}, Адрес - {result[1]}")
                    else:
                        st.write("Информация о номере не найдена. Пожалуйста, введите данные ниже:")

    if st.session_state['plate_number']:
        with st.form(key='new_data_form'):
            st.write(f"Гос. номер: {st.session_state['plate_number']}")
            full_name = st.text_input("ФИО", value=st.session_state['full_name'])
            address = st.text_input("Адрес", value=st.session_state['address'])
            submit_new_data = st.form_submit_button(label='Сохранить')
            if submit_new_data:
                if full_name and address:
                    try:
                        c.execute("INSERT INTO plates (plate_number, full_name, address) VALUES (?, ?, ?)",
                                  (st.session_state['plate_number'], full_name, address))
                        conn.commit()
                        st.success("Информация сохранена")
                        # Reset session state
                        st.session_state['plate_number'] = ''
                        st.session_state['full_name'] = ''
                        st.session_state['address'] = ''
                        st.session_state['save_success'] = True
                    except sqlite3.Error as e:
                        st.error(f"Ошибка при сохранении данных: {e}")
                else:
                    st.error("Пожалуйста, введите все данные.")

    if st.session_state['save_success']:
        st.success("Информация успешно записана в базу данных.")
        st.session_state['save_success'] = False

if __name__ == "__main__":
    app()
