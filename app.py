import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import os
import time
from datetime import datetime
from PIL import Image

st.set_page_config(
    page_title="Night High-Beam Detection & Alert System",
    page_icon="🚨",
    layout="wide"
)

DB_PATH = "traffic_records.db"
EVIDENCE_DIR = "evidence_store"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 1. DATABASE SETUP
# -----------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        event_id TEXT PRIMARY KEY,
        timestamp TEXT,
        camera_id TEXT,
        plate_number TEXT,
        headlight_status TEXT,
        glare_score REAL,
        message_sent_status TEXT,
        full_image_path TEXT,
        status TEXT DEFAULT 'Pending Police Verification'
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS message_notifications (
        notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        recipient TEXT,
        message_payload TEXT,
        dispatch_time TEXT,
        delivery_status TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------------------------------------------
# 2. DISPATCH NOTIFICATION ENGINE (SMS / POLICE API / WEBHOOK)
# -----------------------------------------------------------------
def send_violation_message(event_id, plate, camera_id, glare_val):
    """
    Simulates sending an immediate dispatch message to the regional
    traffic control room or officer SMS terminal.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (f"[ALERT] High-Beam Glare Identified! "
           f"Camera: {camera_id} | Plate: {plate} | Glare Index: {glare_val:.2f} | Time: {timestamp}")
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO message_notifications (event_id, recipient, message_payload, dispatch_time, delivery_status)
    VALUES (?, ?, ?, ?, ?)
    """, (event_id, "Central Traffic Enforcement Unit (Station 3)", msg, timestamp, "SUCCESS: SENT"))
    conn.commit()
    conn.close()
    
    return msg

# -----------------------------------------------------------------
# 3. NIGHTTIME HIGH-BEAM COMPUTER VISION DETECTOR
# -----------------------------------------------------------------
class NightGlareDetector:
    def __init__(self, lum_thresh=220, glare_area_ratio_thresh=0.08):
        self.lum_thresh = lum_thresh
        self.glare_ratio_thresh = glare_area_ratio_thresh

    def analyze_frame(self, frame_bgr):
        """
        Converts to LAB, analyzes light saturation spread in the upper horizon.
        """
        h, w = frame_bgr.shape[:2]
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)

        # Upper beam projection zone (where elevated glare dazzles oncoming traffic)
        upper_zone = l_channel[int(h * 0.25):int(h * 0.75), :]
        
        # Saturated glare pixels
        _, binary_glare = cv2.threshold(upper_zone, self.lum_thresh, 255, cv2.THRESH_BINARY)
        glare_pixels = cv2.countNonZero(binary_glare)
        total_pixels = upper_zone.shape[0] * upper_zone.shape[1]
        
        glare_ratio = glare_pixels / float(total_pixels) if total_pixels > 0 else 0
        is_high_beam = glare_ratio >= self.glare_thresh

        # Find glare contours to place bounding boxes
        contours, _ = cv2.findContours(binary_glare, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for c in contours:
            if cv2.contourArea(c) > 150:
                bx, by, bw, bh = cv2.boundingRect(c)
                # Offset Y back to original frame coordinates
                bounding_boxes.append((bx, by + int(h * 0.25), bw, bh))

        return is_high_beam, round(glare_ratio, 3), bounding_boxes

# -----------------------------------------------------------------
# 4. STREAMLIT APPLICATION UI
# -----------------------------------------------------------------
st.sidebar.title("🚨 Traffic Enforcement")
menu = st.sidebar.radio("Navigation", ["Live Night Detection Stream", "Dispatched Messages Log", "Police Verification Desk"])

detector = NightGlareDetector(lum_thresh=215, glare_area_ratio_thresh=0.05)

if menu == "Live Night Detection Stream":
    st.title("🌙 Real-Time Night High-Beam Detection & Dispatch")
    st.caption("Active monitoring for excessive glare and automated police notification triggers.")

    col_view, col_info = st.columns([2, 1])

    with col_info:
        st.subheader("Inspection Controls")
        camera_id = st.text_input("Active Camera ID", value="CAM-NIGHT-04")
        test_plate = st.text_input("Simulated Vehicle Plate", value="KA05MR8841")
        
        trigger_sim = st.button("Trigger Test High-Beam Vehicle Frame", type="primary")

    with col_view:
        # Create a blank or simulated dark road frame
        frame = np.zeros((400, 700, 3), dtype=np.uint8)
        # Background dark road styling
        frame[:] = (18, 18, 22)

        high_beam_flag = False
        glare_score = 0.0
        boxes = []

        if trigger_sim:
            # Render a vehicle emitting high-beam glaring light
            # Vehicle silhouette
            cv2.rectangle(frame, (180, 150), (520, 340), (45, 55, 65), -1)
            # Dazzling headlights
            cv2.circle(frame, (250, 240), 45, (255, 255, 255), -1)
            cv2.circle(frame, (450, 240), 45, (255, 255, 255), -1)
            # High-beam glare corona
            cv2.ellipse(frame, (250, 240), (80, 50), 0, 0, 360, (210, 255, 255), -1)
            cv2.ellipse(frame, (450, 240), (80, 50), 0, 0, 360, (210, 255, 255), -1)

            # Analyze frame
            high_beam_flag, glare_score, boxes = detector.analyze_frame(frame)

        if high_beam_flag:
            event_id = f"EVT-{int(time.time())}"
            
            # Draw detected glare regions
            for (bx, by, bw, bh) in boxes:
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)

            # Draw visual banners on frame
            cv2.rectangle(frame, (10, 10), (480, 75), (0, 0, 180), -1)
            cv2.putText(frame, "HIGH BEAM DETECTED", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "STATUS: MESSAGE SENT TO POLICE", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Save full image capture
            img_path = os.path.join(EVIDENCE_DIR, f"{event_id}.jpg")
            cv2.imwrite(img_path, frame)

            # Save incident record
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO incidents (event_id, timestamp, camera_id, plate_number, headlight_status, glare_score, message_sent_status, full_image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera_id, test_plate, 
                  "High Beam Glare", glare_score, "DISPATCHED", img_path))
            conn.commit()
            conn.close()

            # Dispatch notification
            dispatch_payload = send_violation_message(event_id, test_plate, camera_id, glare_score)
            
            st.error("🚨 HIGH BEAM DETECTED!")
            st.success(f"📩 {dispatch_payload}")
        else:
            cv2.putText(frame, "NORMAL NIGHT MONITORING (NO HIGH BEAM)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Render display
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

elif menu == "Dispatched Messages Log":
    st.title("📩 Automated Dispatched Message Log")
    st.write("Real-time notifications pushed to police control rooms and patrol units.")

    conn = sqlite3.connect(DB_PATH)
    msg_df = pd.read_sql("SELECT * FROM message_notifications ORDER BY notification_id DESC", conn)
    conn.close()

    if msg_df.empty:
        st.info("No dispatched violation messages yet. Trigger an event from the live stream.")
    else:
        st.dataframe(msg_df, use_container_width=True)

elif menu == "Police Verification Desk":
    st.title("👮 Police Decision Support & Verification Desk")
    st.caption("Legal Requirement: AI detection triggers alerts; authorized officers must verify and confirm before penalties are issued.")

    conn = sqlite3.connect(DB_PATH)
    pending_df = pd.read_sql("SELECT * FROM incidents ORDER BY timestamp DESC", conn)
    conn.close()

    if pending_df.empty:
        st.info("No incidents awaiting adjudication.")
    else:
        st.dataframe(pending_df[['event_id', 'timestamp', 'camera_id', 'plate_number', 'glare_score', 'status', 'message_sent_status']], use_container_width=True)
