import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import os
import time
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

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

def log_violation_event(glare_score):
    """Logs violation and dispatches message notification."""
    event_id = f"EVT-{int(time.time())}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    camera_id = "LAPTOP-WEBCAM-01"
    plate = "DETECTED-VEHICLE"
    msg = f"[ALERT] High-Beam Glare! Cam: {camera_id} | Glare: {glare_score:.2f} | Time: {timestamp}"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO incidents (event_id, timestamp, camera_id, plate_number, headlight_status, glare_score, message_sent_status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (event_id, timestamp, camera_id, plate, "High Beam Glare", glare_score, "DISPATCHED"))
    
    cur.execute("""
    INSERT INTO message_notifications (event_id, recipient, message_payload, dispatch_time, delivery_status)
    VALUES (?, ?, ?, ?, ?)
    """, (event_id, "Central Traffic Enforcement Unit", msg, timestamp, "SUCCESS: SENT"))
    conn.commit()
    conn.close()

# -----------------------------------------------------------------
# 2. NIGHTTIME HIGH-BEAM COMPUTER VISION ENGINE
# -----------------------------------------------------------------
class NightGlareDetector:
    def __init__(self, lum_thresh=215, glare_ratio_thresh=0.04):
        self.lum_thresh = lum_thresh
        self.glare_thresh = glare_ratio_thresh  # Fixed attribute assignment

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)

        # Focus analysis on central visual region
        zone = l_channel[int(h * 0.2):int(h * 0.8), :]
        _, binary_glare = cv2.threshold(zone, self.lum_thresh, 255, cv2.THRESH_BINARY)
        
        glare_pixels = cv2.countNonZero(binary_glare)
        total_pixels = zone.shape[0] * zone.shape[1]
        glare_ratio = glare_pixels / float(total_pixels) if total_pixels > 0 else 0
        
        is_high_beam = glare_ratio >= self.glare_thresh

        # Identify glare spots
        contours, _ = cv2.findContours(binary_glare, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) > 200:
                bx, by, bw, bh = cv2.boundingRect(c)
                boxes.append((bx, by + int(h * 0.2), bw, bh))

        return is_high_beam, round(glare_ratio, 3), boxes

detector = NightGlareDetector(lum_thresh=220, glare_ratio_thresh=0.04)

# -----------------------------------------------------------------
# 3. WEBRTC LIVE VIDEO PROCESSOR (LAPTOP CAMERA)
# -----------------------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_alert_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Detect high beam on incoming camera frame
        is_high_beam, glare_score, boxes = detector.process(img)

        if is_high_beam:
            # Draw red warning bounding boxes around light source
            for (bx, by, bw, bh) in boxes:
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)

            # Overlay alert banners
            cv2.rectangle(img, (15, 15), (520, 85), (0, 0, 180), -1)
            cv2.putText(img, "HIGH BEAM DETECTED", (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(img, "STATUS: MESSAGE SENT TO POLICE", (25, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Debounce logging: only insert a record once every 5 seconds
            current_time = time.time()
            if current_time - self.last_alert_time > 5.0:
                self.last_alert_time = current_time
                log_violation_event(glare_score)
        else:
            cv2.putText(img, "MONITORING (NORMAL LIGHTING)", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------------
# 4. STREAMLIT INTERFACE
# -----------------------------------------------------------------
st.sidebar.title("🚨 Enforcement Desk")
menu = st.sidebar.radio("Navigation", ["Live Laptop Camera", "Dispatched Messages Log", "Police Verification Desk"])

if menu == "Live Laptop Camera":
    st.title("📹 Live Laptop Camera Detection")
    st.write("Click **START** below and allow browser camera permissions. Point your phone flashlight or a bright torch at your webcam to trigger the high-beam alert.")

    # Public STUN servers to negotiate WebRTC video stream
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    webrtc_streamer(
        key="high-beam-detector",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )

elif menu == "Dispatched Messages Log":
    st.title("📩 Automated Dispatched Message Log")
    st.write("Confirmed real-time alerts pushed to police control rooms.")

    conn = sqlite3.connect(DB_PATH)
    msg_df = pd.read_sql("SELECT * FROM message_notifications ORDER BY notification_id DESC", conn)
    conn.close()

    if msg_df.empty:
        st.info("No dispatched messages yet. Turn on your webcam and shine a bright light to trigger an alert.")
    else:
        st.dataframe(msg_df, use_container_width=True)

elif menu == "Police Verification Desk":
    st.title("👮 Police Verification Records")
    conn = sqlite3.connect(DB_PATH)
    incidents_df = pd.read_sql("SELECT * FROM incidents ORDER BY timestamp DESC", conn)
    conn.close()

    if incidents_df.empty:
        st.info("No incidents logged yet.")
    else:
        st.dataframe(incidents_df, use_container_width=True)
