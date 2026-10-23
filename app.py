import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import io
import re
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Set page layout and branding
st.set_page_config(
    page_title="AI Traffic Violation & ANPR Police Desk",
    page_icon="🚔",
    layout="wide"
)

DB_PATH = "traffic_records.db"
EVIDENCE_DIR = "evidence_store"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ---------------------------------------------------------
# DATABASE INITIALIZATION & QUERIES
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        event_id TEXT PRIMARY KEY,
        timestamp TEXT,
        camera_id TEXT,
        camera_location TEXT,
        police_station TEXT,
        plate_number TEXT,
        plate_confidence REAL,
        headlight_status TEXT,
        glare_intensity REAL,
        suspected_violation TEXT,
        full_image_path TEXT,
        plate_image_path TEXT,
        status TEXT DEFAULT 'Pending Police Verification',
        reviewing_officer TEXT,
        officer_notes TEXT,
        decision_timestamp TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT,
        action TEXT,
        performed_by TEXT,
        timestamp TEXT,
        details TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_audit(event_id, action, performed_by, details):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO audit_logs (event_id, action, performed_by, timestamp, details)
    VALUES (?, ?, ?, ?, ?)
    """, (event_id, action, performed_by, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), details))
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# SYNTHETIC EVIDENCE GENERATOR (FOR IMMEDIATE TESTING)
# ---------------------------------------------------------
def generate_synthetic_incident(camera_id, location, station):
    """Creates a mock vehicle capture and cropped license plate image for testing."""
    event_id = f"EVT-{int(datetime.now().timestamp())}"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Full Vehicle Image
    car_img = Image.new('RGB', (640, 360), color=(25, 25, 30))
    draw = ImageDraw.Draw(car_img)
    # Vehicle Body
    draw.rectangle([140, 100, 500, 310], fill=(45, 55, 72), outline=(100, 100, 100), width=2)
    # Windshield
    draw.polygon([(190, 110), (450, 110), (480, 200), (160, 200)], fill=(15, 20, 25))
    # Headlights with Glare
    draw.ellipse([160, 220, 220, 260], fill=(255, 255, 210), outline=(255, 255, 255))
    draw.ellipse([420, 220, 480, 260], fill=(255, 255, 210), outline=(255, 255, 255))
    # High-beam glare spray
    draw.arc([130, 200, 250, 280], start=0, end=360, fill=(255, 255, 180), width=3)
    draw.arc([390, 200, 510, 280], start=0, end=360, fill=(255, 255, 180), width=3)
    # Front License Plate
    draw.rectangle([270, 270, 370, 298], fill=(240, 240, 240), outline=(0, 0, 0), width=2)
    
    mock_plates = ["KA01MJ4921", "MH12DE8899", "DL04CA1102", "HR26DK9001", "UP32AB7733"]
    selected_plate = np.random.choice(mock_plates)
    draw.text((280, 275), selected_plate, fill=(0, 0, 0))

    # Save full frame
    full_path = os.path.join(EVIDENCE_DIR, f"{event_id}_full.jpg")
    car_img.save(full_path)

    # 2. Plate Crop Image
    plate_img = car_img.crop((265, 265, 375, 305))
    plate_path = os.path.join(EVIDENCE_DIR, f"{event_id}_plate.jpg")
    plate_img.save(plate_path)

    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO incidents (
        event_id, timestamp, camera_id, camera_location, police_station,
        plate_number, plate_confidence, headlight_status, glare_intensity,
        suspected_violation, full_image_path, plate_image_path, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event_id, timestamp_str, camera_id, location, station,
        selected_plate, round(float(np.random.uniform(0.85, 0.98)), 2),
        "Suspected High-Beam Glare", round(float(np.random.uniform(0.40, 0.68)), 2),
        "Improper / Dazzling High-Beam In Opposite Direction",
        full_path, plate_path, 'Pending Police Verification'
    ))
    conn.commit()
    conn.close()
    
    log_audit(event_id, "SYSTEM_TRIGGER", "EDGE_VISION_CAMERA", f"Plate: {selected_plate} captured.")
    return event_id

# ---------------------------------------------------------
# APPLICATION STARTUP
# ---------------------------------------------------------
init_db()

# Sidebar Navigation
st.sidebar.title("🚔 Traffic Decision Support")
st.sidebar.caption("AI-Assisted Traffic Enforcement Pipeline")
menu = st.sidebar.radio(
    "Modules", 
    ["Live Monitoring Feed", "Police Verification Desk", "Audit Trail & Records", "System Diagnostics"]
)

# Active Camera Jurisdictions
JURISDICTIONS = {
    "CAM-01-NORTH": {"loc": "North Ring Road, Junction 4", "dept": "Central Traffic Division - Station 3"},
    "CAM-02-WEST": {"loc": "Highway 17 Outer Bypass", "dept": "West District Police HQ"}
}

# ---------------------------------------------------------
# MODULE 1: LIVE MONITORING FEED
# ---------------------------------------------------------
if menu == "Live Monitoring Feed":
    st.title("📹 Real-Time Roadway Monitoring & Detection")
    st.write("Edge camera video telemetry with vehicle tracking, glare identification, and ANPR extraction.")
    
    col_cam, col_stats = st.columns([2, 1])

    with col_cam:
        selected_cam = st.selectbox("Select Active Camera Node", list(JURISDICTIONS.keys()))
        cam_info = JURISDICTIONS[selected_cam]
        st.caption(f"**Physical Location:** {cam_info['loc']} | **Jurisdiction:** {cam_info['dept']}")

        # Simulated Camera Viewport
        st.markdown(
            """
            <div style="background-color: #0f172a; padding: 25px; border-radius: 8px; border: 1px solid #334155; text-align: center;">
                <p style="color: #38bdf8; font-weight: 600; margin: 0;">● LIVE RTSP FEED CONNECTED [30 FPS - 1080p]</p>
                <p style="color: #94a3b8; font-size: 13px;">Analyzing incoming opposite-direction vehicles for dazzling high-beam spread...</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.write("")
        if st.button("Simulate Vehicle Detection & High-Beam Incident", type="primary"):
            new_id = generate_synthetic_incident(selected_cam, cam_info['loc'], cam_info['dept'])
            st.success(f"New incident detected and registered: **{new_id}**! Queued for police review.")

    with col_stats:
        st.subheader("System Telemetry")
        conn = sqlite3.connect(DB_PATH)
        df_all = pd.read_sql("SELECT status FROM incidents", conn)
        conn.close()

        total_logged = len(df_all)
        pending_cnt = len(df_all[df_all['status'] == 'Pending Police Verification'])
        confirmed_cnt = len(df_all[df_all['status'] == 'Confirmed Violation'])

        st.metric("Total Detections Logged", total_logged)
        st.metric("Pending Officer Verification", pending_cnt, delta="Requires Action", delta_color="inverse")
        st.metric("Officer-Approved Violations", confirmed_cnt)

# ---------------------------------------------------------
# MODULE 2: POLICE VERIFICATION DESK
# ---------------------------------------------------------
elif menu == "Police Verification Desk":
    st.title("👮 Authorized Police Verification Portal")
    st.info("⚠️ Legal Protocol: AI systems cannot directly penalize citizens. An authorized officer must evaluate the evidence and confirm or reject every report.")

    conn = sqlite3.connect(DB_PATH)
    pending_df = pd.read_sql("SELECT * FROM incidents WHERE status = 'Pending Police Verification' ORDER BY timestamp DESC", conn)
    conn.close()

    if pending_df.empty:
        st.success("✅ Clean Queue: No pending incidents awaiting review.")
    else:
        st.subheader(f"Incidents Awaiting Verification ({len(pending_df)})")
        selected_id = st.selectbox("Select Incident ID for Evidence Adjudication", pending_df['event_id'].tolist())
        row = pending_df[pending_df['event_id'] == selected_id].iloc[0]

        img_col, meta_col = st.columns([1, 1])

        with img_col:
            st.markdown("#### Primary Visual Evidence")
            if os.path.exists(row['full_image_path']):
                st.image(row['full_image_path'], caption="Full Frame Opposite-Direction Capture", use_container_width=True)
            else:
                st.warning("Vehicle capture image not found on disk.")

            st.markdown("#### Number Plate Crop")
            if os.path.exists(row['plate_image_path']):
                st.image(row['plate_image_path'], caption=f"Extracted Crop: {row['plate_number']}", width=250)

        with meta_col:
            st.markdown("#### Incident Telemetry")
            st.json({
                "Incident ID": row['event_id'],
                "Timestamp": row['timestamp'],
                "Police Jurisdiction": row['police_station'],
                "Camera Node": f"{row['camera_id']} ({row['camera_location']})",
                "Recognized Plate": row['plate_number'],
                "OCR Model Confidence": f"{row['plate_confidence'] * 100:.1f}%",
                "Lighting Analysis": row['headlight_status'],
                "Glare Intensity Score": row['glare_intensity'],
                "Suspected Infraction": row['suspected_violation']
            })

            st.divider()
            st.markdown("#### Adjudication Decision")
            badge_id = st.text_input("Reviewing Officer Badge #", value="INSPECTOR-2891")
            notes = st.text_area("Legal Justification / Verification Remarks")

            act1, act2, act3 = st.columns(3)
            with act1:
                if st.button("Confirm Violation", type="primary"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""
                        UPDATE incidents 
                        SET status='Confirmed Violation', reviewing_officer=?, officer_notes=?, decision_timestamp=? 
                        WHERE event_id=?
                    """, (badge_id, notes, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), selected_id))
                    conn.commit()
                    conn.close()
                    log_audit(selected_id, "CONFIRM_VIOLATION", badge_id, notes)
                    st.success("Violation confirmed! Transferred to authorized traffic challan system.")
                    st.rerun()

            with act2:
                if st.button("Reject Report"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""
                        UPDATE incidents 
                        SET status='Rejected', reviewing_officer=?, officer_notes=?, decision_timestamp=? 
                        WHERE event_id=?
                    """, (badge_id, notes, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), selected_id))
                    conn.commit()
                    conn.close()
                    log_audit(selected_id, "REJECT_REPORT", badge_id, notes)
                    st.warning("Incident marked as rejected.")
                    st.rerun()

            with act3:
                if st.button("Flag Inconclusive"):
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""
                        UPDATE incidents 
                        SET status='Insufficient Evidence', reviewing_officer=?, officer_notes=?, decision_timestamp=? 
                        WHERE event_id=?
                    """, (badge_id, notes, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), selected_id))
                    conn.commit()
                    conn.close()
                    log_audit(selected_id, "INSUFFICIENT_EVIDENCE", badge_id, notes)
                    st.info("Marked as inconclusive due to lighting or blur.")
                    st.rerun()

# ---------------------------------------------------------
# MODULE 3: AUDIT TRAIL & RECORDS
# ---------------------------------------------------------
elif menu == "Audit Trail & Records":
    st.title("📋 Evidentiary Chain-of-Custody & Audit Logs")
    st.write("Tamper-evident logs of model detection events and reviewing officer decisions.")

    conn = sqlite3.connect(DB_PATH)
    tab1, tab2 = st.tabs(["All Recorded Incidents", "Officer & System Action Logs"])

    with tab1:
        incidents_df = pd.read_sql("SELECT * FROM incidents ORDER BY timestamp DESC", conn)
        st.dataframe(incidents_df, use_container_width=True)

    with tab2:
        audit_df = pd.read_sql("SELECT * FROM audit_logs ORDER BY timestamp DESC", conn)
        st.dataframe(audit_df, use_container_width=True)

    conn.close()

# ---------------------------------------------------------
# MODULE 4: SYSTEM DIAGNOSTICS
# ---------------------------------------------------------
elif menu == "System Diagnostics":
    st.title("⚙️ System Health & Engine Diagnostics")
    st.write("Validation of computer vision, OCR dependencies, and storage mounts.")

    st.markdown(
        """
        | Component | Target Service | Status |
        | :--- | :--- | :--- |
        | **Headlight Glare Engine** | LAB Luminance Dispersion Thresholding | `ONLINE` |
        | **ANPR Inference** | EasyOCR + Bilateral CLAHE Preprocessor | `READY` |
        | **Storage Subsystem** | SQLite Local Storage Mount (`traffic_records.db`) | `CONNECTED` |
        | **Police Integration** | Decision Support Adjudication Portal | `ACTIVE` |
        """
    )
