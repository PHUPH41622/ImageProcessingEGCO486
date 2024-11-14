from typing import List, Dict, Any, Tuple
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime, timedelta
import pandas as pd
import base64
from PIL import Image
import yaml
import time

# Load model and config functions
def read_yaml(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data: Dict = yaml.safe_load(file)
    return data

def get_price_from_index(index_result: int) -> int:
    price_mapping = {
        0: 10,  # snack
        1: 7,   # water
        2: 12,  # milk
        3: 20,  # crackers
        4: 15   # candy
    }
    return price_mapping.get(index_result, 0)

# Encode an image array to a base64 string
def encode_image(image_array):
    """Convert image array to base64 string for storage."""
    success, encoded_image = cv2.imencode('.png', image_array)
    if success:
        return base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    return None

# Decode a base64 string back to an image array
def decode_image(base64_string):
    """Convert base64 string back to image array."""
    if base64_string:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return None

# Calculate the receipt details from detected items
def calculate_receipt_details(detected_items: List[Dict]) -> Dict:
    """Calculate receipt details from detected items."""
    item_summary = {}
    total = 0

    for item in detected_items:
        name = item['name']
        price = item['price']

        if name in item_summary:
            item_summary[name]['count'] += 1
            item_summary[name]['subtotal'] += price
        else:
            item_summary[name] = {
                'count': 1,
                'price': price,
                'subtotal': price
            }
        total += price

    return {
        'items': item_summary,
        'total': total
    }

# Generate a unique receipt number based on timestamp
def generate_receipt_number() -> str:
    """Generate receipt number based on timestamp."""
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")

# Save receipt details into a CSV file
def save_receipt(receipt_number: str, timestamp: str, receipt_details: Dict, processed_image: np.ndarray):
    """Save receipt details to a CSV file."""
    items_str = '; '.join([f"{name}: {details['count']} pcs (price: {details['price']} ฿)" 
                          for name, details in receipt_details['items'].items()])
    
    image_b64 = encode_image(processed_image)
    
    receipt_data = {
        'receipt_number': receipt_number,
        'timestamp': timestamp,
        'items': items_str,
        'total': receipt_details['total'],
        'processed_image': image_b64
    }
    
    try:
        df = pd.read_csv('receipts.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['receipt_number', 'timestamp', 'items', 'total', 'processed_image'])
    
    df = pd.concat([df, pd.DataFrame([receipt_data])], ignore_index=True)
    df.to_csv('receipts.csv', index=False)

# Display receipt history with pagination
def display_receipt_history(df: pd.DataFrame, page: int, items_per_page: int):
    """Display receipt history with pagination and expandable details."""
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)

        total_pages = (len(df) + items_per_page - 1) // items_per_page
        start_idx = page * items_per_page
        end_idx = min(start_idx + items_per_page, len(df))

        for idx in range(start_idx, end_idx):
            receipt = df.iloc[idx]
            with st.expander(
                f"Receipt No: {receipt['receipt_number']} | "
                f"Date: {receipt['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Total: {receipt['total']} ฿"
            ):
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.write("### Receipt Details")
                    st.write(f"**Date and Time:** {receipt['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write("### Product List")
                    items_list = receipt['items'].split('; ')
                    for item in items_list:
                        st.write(f"- {item}")

                with col2:
                    st.write("### Detected Product Image")
                    img_array = decode_image(receipt['processed_image'])
                    if img_array is not None:
                        st.image(img_array, use_column_width=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if page > 0:
                if st.button("← Previous"):
                    st.session_state.page = page - 1
                    st.rerun()

        with col2:
            st.write(f"Page {page + 1} of {total_pages}")

        with col3:
            if page < total_pages - 1:
                if st.button("Next →"):
                    st.session_state.page = page + 1
                    st.rerun()
    else:
        st.info("No receipt history available.")

# Helper function to load receipts from a CSV file and filter by date range
def load_receipts(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load and filter receipt data by date range."""
    try:
        df = pd.read_csv('receipts.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        return df[mask]
    except FileNotFoundError:
        return pd.DataFrame(columns=['receipt_number', 'timestamp', 'items', 'total', 'processed_image'])



def map_result_to_price(result: List) -> int:
    return sum(get_price_from_index(index) for index in result)

# Detection function
def detect_objects(image: np.ndarray, model: YOLO, threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
    results = model(image)[0]
    result_list = []
    detected_items = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            class_id = int(class_id)
            result_list.append(class_id)
            
            item_name = ["Snack", "Water", "Milk", "Crackers", "Candy"][class_id]
            detected_items.append({
                'class_id': class_id,
                'name': item_name.upper(),
                'price': get_price_from_index(class_id),
                'box': (int(x1), int(y1), int(x2), int(y2))
            })
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            cv2.putText(image, item_name.upper(), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3)

    total_price = map_result_to_price(result_list)
    cv2.putText(image, f"Total Price: {total_price}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detected_items

# Load YOLO model
@st.cache_resource
def load_model():
   model_path = os.path.join('.', 'weights', 'yolov11n_aug.pt')
   return YOLO(model_path)

import time  # เพิ่มการนำเข้า time

def main():
    st.set_page_config(layout="wide")
    model = load_model()
    st.title("Product Scanning and Receipt System")
    
    left_col, right_col = st.columns([1, 1])

    # ตรวจสอบและตั้งค่าเริ่มต้นใน session_state
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'capture_image' not in st.session_state:
        st.session_state.capture_image = False
    if 'capture_button_counter' not in st.session_state:
        st.session_state.capture_button_counter = 0
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    # คอลัมน์ซ้าย: สำหรับการอัปโหลดรูปภาพหรือการจับภาพจากกล้อง
    with left_col:
        st.subheader("Upload or Capture an Image")
        
        # ตัวเลือกการอัปโหลดรูปภาพ
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

        threshold = st.slider("Detection Threshold", 0.1, 1.0, 0.5)
        
        # ปุ่มเปิด/ปิดกล้อง
        if st.button("Open Camera" if not st.session_state.camera_on else "Close Camera", key="toggle_camera"):
            st.session_state.camera_on = not st.session_state.camera_on
            # เปิดการจับภาพจากกล้องถ้าเปิดกล้อง
            if st.session_state.camera_on:
                st.session_state.cap = cv2.VideoCapture(0)
            else:
                if st.session_state.cap:
                    st.session_state.cap.release()
                    st.session_state.cap = None

        # การแสดงผลจากกล้อง
        if st.session_state.camera_on and st.session_state.cap is not None:
            # ใช้ st.empty() เพื่อสร้างตำแหน่งว่างที่สามารถอัปเดตได้
            frame_placeholder = st.empty()
            
            # อ่านเฟรมจากกล้องและแสดงผล
            ret, frame = st.session_state.cap.read()
            if ret:
                processed_frame, detected_items = detect_objects(frame.copy(), model, threshold)
                frame_placeholder.image(processed_frame, channels="RGB")
                
            # ใช้การรีเฟรชหน้าเพื่อให้เฟรมอัปเดตใหม่
            time.sleep(0.05)
            st.experimental_rerun()

        # ประมวลผลรูปภาพที่อัปโหลดหรือจับภาพ
        if uploaded_file is not None or st.session_state.capture_image:
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
            elif st.session_state.capture_image:
                if st.session_state.cap is None:
                    st.error("Camera capture failed. Please try again.")
                    return
                ret, image_array = st.session_state.cap.read()
                if not ret or image_array is None:
                    st.error("Failed to capture image from camera.")
                    return
                image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

            processed_image, detected_items = detect_objects(image_array, model, threshold)
            st.image(processed_image, caption='Detected Products', use_column_width=True)

            if detected_items:
                receipt_details = calculate_receipt_details(detected_items)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                receipt_number = generate_receipt_number()

                st.subheader("Receipt")
                st.write(f"Receipt Number: {receipt_number}")
                st.write(f"Date and Time: {timestamp}")

                st.write("Product List:")
                for item_name, details in receipt_details['items'].items():
                    st.write(f"- {item_name}: {details['count']} pcs (Price: {details['price']} ฿, Subtotal: {details['subtotal']} ฿)")
                
                st.write(f"Total Price: {receipt_details['total']} ฿")

                if st.button("Save Receipt", key="save_receipt"):
                    save_receipt(receipt_number, timestamp, receipt_details, processed_image)
                    st.success(f"Receipt {receipt_number} saved successfully.")
                    st.session_state.capture_image = False

    # คอลัมน์ขวา: แสดงผลประวัติใบเสร็จ (ไม่ขึ้นกับสถานะของกล้อง)
    with right_col:
        st.title("Receipt History")
        
        # การตั้งค่าการแบ่งหน้า
        if 'page' not in st.session_state:
            st.session_state.page = 0

        # ตัวกรองวันที่
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        end_date = st.date_input("End Date", datetime.now())

        # โหลดข้อมูลใบเสร็จและกรองตามช่วงวันที่
        df = load_receipts(start_date, end_date)
        
        # แสดงผลประวัติใบเสร็จ พร้อมระบบแบ่งหน้า
        items_per_page = 5
        display_receipt_history(df, st.session_state.page, items_per_page)

if __name__ == '__main__':
    main()








