import cv2
import os
import numpy as np
from utils import map_result_to_price
from ultralytics import YOLO
import streamlit as st
import time

# โหลดโมเดล YOLO
model_path = os.path.join('.', 'weights', 'yolov11n_aug.pt')
model = YOLO(model_path)

# ตั้งค่าหน้า Streamlit
st.title("Real-Time Object Detection with Camera")
st.write("กดปุ่มเพื่อเปิด/ปิดกล้องและจับภาพเมื่อมีการตรวจจับวัตถุ")

# ตัวเลื่อนสำหรับปรับค่า Threshold
threshold = st.slider("Threshold", min_value=0.1, max_value=1.0, value=0.5)

# ตั้งค่า session_state สำหรับสถานะของกล้องและการจับภาพ
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'capture_image' not in st.session_state:
    st.session_state.capture_image = False

# ปุ่มเปิด/ปิดกล้อง
if st.button("เปิดกล้อง" if not st.session_state.camera_on else "ปิดกล้อง"):
    st.session_state.camera_on = not st.session_state.camera_on

# ปุ่มแคปรูป
if st.button("จับภาพ"):
    st.session_state.capture_image = True

# ใช้ st.empty() เพื่อสร้างตำแหน่งสำหรับแสดงภาพแบบสด
frame_placeholder = st.empty()

# สร้างโฟลเดอร์สำหรับบันทึกภาพ (ถ้ายังไม่มี)
os.makedirs("captured_images", exist_ok=True)

# ตรวจสอบสถานะการเปิด/ปิดกล้อง
cap = None
if st.session_state.camera_on:
    # เปิดการเชื่อมต่อกับกล้อง
    if cap is None:
        cap = cv2.VideoCapture(0)
    while st.session_state.camera_on and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("ไม่สามารถเข้าถึงกล้องได้")
            break

        # ตรวจจับวัตถุในเฟรม
        results = model(frame)[0]
        result_list = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            # กำหนดชื่อสินค้า
            if class_id == 0:
                item_name = "Snack"
            elif class_id == 1:
                item_name = "Water"
            elif class_id == 2:
                item_name = "Milk"
            elif class_id == 3:
                item_name = "Crackers"
            elif class_id == 4:
                item_name = "Candy"

            # ตรวจสอบว่าความเชื่อมั่นสูงกว่าค่า threshold
            if score > threshold:
                # วาดกรอบสี่เหลี่ยมและแสดงชื่อสินค้า
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, item_name.upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            result_list.append(int(class_id))

        # คำนวณราคาและแสดงผลรวมบนเฟรม
        total_price = map_result_to_price(result_list)
        cv2.putText(frame, f"Total Price: {total_price}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # แปลงเฟรมจาก BGR เป็น RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # แสดงเฟรมใน Streamlit โดยอัปเดตตำแหน่งภาพ
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # บันทึกภาพเมื่อกดปุ่มจับภาพ และตั้งค่าสถานะให้กลับเป็น False หลังจากจับภาพ
        if st.session_state.capture_image:
            # ปรับค่าความสว่าง
            adjusted_frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)  # ปรับค่า alpha และ beta ตามที่เหมาะสม
            capture_path = f"captured_images/capture_{time.strftime('%Y%m%d-%H%M%S')}.png"
            cv2.imwrite(capture_path, adjusted_frame)  # ใช้ภาพที่ปรับแสงแล้วในการบันทึก
            st.success(f"ภาพถูกบันทึกไว้ที่ {capture_path}")
            st.session_state.capture_image = False  # รีเซ็ตสถานะการจับภาพ

        # เพิ่ม delay เล็กน้อยเพื่อลดการใช้งาน CPU
        time.sleep(0.05)

    # ปิดกล้องเมื่อ session_state.camera_on เป็น False
    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("กล้องปิดอยู่ กรุณากดปุ่มเปิดกล้องเพื่อเริ่มการตรวจจับ")