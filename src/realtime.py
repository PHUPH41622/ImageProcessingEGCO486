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

def read_yaml(file_path: str) -> Dict:
    """Get the key, value in the config."""
    with open(file_path, 'r') as file:
        data: Dict = yaml.safe_load(file)
    return data

def get_price_from_index(index_result: int) -> int:
    """Get the price from the index result."""
    price_mapping = {
        0: 10,  # snack
        1: 7,   # water
        2: 12,  # milk
        3: 20,  # crackers
        4: 15   # candy
    }
    return price_mapping.get(index_result, 0)

def map_result_to_price(result: List) -> int:
    """Map the result to the price."""
    return sum(get_price_from_index(index) for index in result)

def load_image(image_file):
    """Load image in BGR format like cv2.imread"""
    img = Image.open(image_file)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_bgr

def encode_image(image_array):
    """Convert image array to base64 string for storage"""
    success, encoded_image = cv2.imencode('.png', image_array)
    if success:
        return base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    return None

def decode_image(base64_string):
    """Convert base64 string back to image array"""
    if base64_string:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return None

def detect_objects(image: np.ndarray, model: YOLO, threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
    """Perform object detection on the image."""
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
            cv2.putText(image, item_name.upper(), (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, image.shape[1]/1000, (255, 0, 0), int(image.shape[1]/1000), cv2.LINE_AA)
    
    total_price = map_result_to_price(result_list)
    cv2.putText(image, f"Total Price: {total_price}", (0, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, image.shape[1]/300, (0, 0, 0), int(image.shape[1]/300), cv2.LINE_AA)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, detected_items

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

def generate_receipt_number() -> str:
    """Generate receipt number based on timestamp."""
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")

def save_receipt(receipt_number: str, timestamp: str, receipt_details: Dict, processed_image: np.ndarray):
    """Save receipt details to CSV file."""
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

def main():
    st.set_page_config(layout="wide")
    
    if 'page' not in st.session_state:
        st.session_state.page = 0
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'capture_image' not in st.session_state:
        st.session_state.capture_image = False
    
    model = YOLO('./weights/yolov11s_aug.pt')

    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.title("Product Scanning and Receipt System")

        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

        if st.button("เปิดกล้อง" if not st.session_state.camera_on else "ปิดกล้อง"):
            st.session_state.camera_on = not st.session_state.camera_on

        if st.button("จับภาพ") and st.session_state.camera_on:
            st.session_state.capture_image = True

        frame_placeholder = st.empty()
        captured_frame = None
        
        cap = None
        if st.session_state.camera_on:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000)
            cap.set(cv2.CAP_PROP_EXPOSURE, -4)
            while st.session_state.camera_on and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("ไม่สามารถเข้าถึงกล้องได้")
                    break

                # Detect and display
                processed_frame, detected_items = detect_objects(frame.copy(), model, threshold=0.5)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_rgb[:, :, 0] = cv2.add(processed_frame[:, :, 0], 20)  # Red channel
                frame_rgb[:, :, 1] = cv2.add(processed_frame[:, :, 1], 10)  # Green channel
                frame_rgb[:, :, 2] = cv2.add(processed_frame[:, :, 2], 0)   # Blue channel
                captured_frame = frame_rgb.copy()

                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                if st.session_state.capture_image:
                    # captured_frame = processed_frame.copy()  # Copy processed frame for capture
                    # captured_frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
                    # captured_frame_rgb[:, :, 0] = cv2.add(captured_frame[:, :, 0], 20)  # Red channel adjustment
                    # captured_frame_rgb[:, :, 1] = cv2.add(captured_frame[:, :, 1], 10)  # Green channel adjustment
                    # captured_frame_rgb[:, :, 2] = cv2.add(captured_frame[:, :, 2], 0)   # Blue channel adjustment
                    # captured_frame_rgb = cv2.convertScaleAbs(captured_frame_rgb, alpha=1.5, beta=50)

                    # # Ensure the image is bright enough: increase brightness/contrast if needed
                    # # captured_frame_rgb = cv2.convertScaleAbs(captured_frame_rgb, alpha=1.5, beta=50)

                    # # Show the captured image
                    st.image(captured_frame, caption="Captured Image", use_column_width=True)

                    st.session_state.capture_image = False
                    # cap.release()
                    st.session_state.camera_on = False
                    # break

                time.sleep(0.05)

            cap.release()

            if detected_items:
                receipt_details = calculate_receipt_details(detected_items)
                st.subheader("Receipt")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                receipt_number = generate_receipt_number()
                st.write(f"Receipt Number: {receipt_number}")
                st.write(f"Date and Time: {timestamp}")
                
                st.write("Product List:")
                for item_name, details in receipt_details['items'].items():
                    st.write(f"- {item_name}: {details['count']} pcs (Price: {details['price']} ฿, Subtotal: {details['subtotal']} ฿)")
                
                st.write(f"Total Price: {receipt_details['total']} ฿")
                
                if st.button("Save Receipt"):
                    save_receipt(receipt_number, timestamp, receipt_details, processed_image)
                    st.success(f"Receipt {receipt_number} saved successfully.")
                    st.rerun()
            else:
                st.warning("No products detected in the image.")

        elif uploaded_file is not None:
            image = load_image(uploaded_file)
            processed_image, detected_items = detect_objects(image.copy(), model)
            st.image(processed_image, caption='Detected Products', use_column_width=True)

            if detected_items:
                receipt_details = calculate_receipt_details(detected_items)
                st.subheader("Receipt")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                receipt_number = generate_receipt_number()
                st.write(f"Receipt Number: {receipt_number}")
                st.write(f"Date and Time: {timestamp}")
                
                st.write("Product List:")
                for item_name, details in receipt_details['items'].items():
                    st.write(f"- {item_name}: {details['count']} pcs (Price: {details['price']} ฿, Subtotal: {details['subtotal']} ฿)")
                
                st.write(f"Total Price: {receipt_details['total']} ฿")
                
                if st.button("Save Receipt"):
                    save_receipt(receipt_number, timestamp, receipt_details, processed_image)
                    st.success(f"Receipt {receipt_number} saved successfully.")
                    st.rerun()
            else:
                st.warning("No products detected in the image.")

    with right_col:
        st.title("Receipt History")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        try:
            df = pd.read_csv('receipts.csv')
            if len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
                filtered_df = df[mask]
                
                if len(filtered_df) > 0:
                    display_receipt_history(filtered_df, st.session_state.page, items_per_page=5)
                else:
                    st.info("No receipts found in the selected date range.")
            else:
                st.info("No receipt history available.")
        except FileNotFoundError:
            st.info("No receipt history available.")

if __name__ == '__main__':
    main()