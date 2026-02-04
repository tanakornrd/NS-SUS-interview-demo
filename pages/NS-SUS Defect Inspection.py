import streamlit as st
import os
import sys
import subprocess
import time
import random

# --- 0. FORCE UPDATE SYSTEM ---
try:
    import google.generativeai as genai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai>=0.8.3"])
    st.rerun()

import google.generativeai as genai
from PIL import Image
import csv
import pandas as pd
import datetime

# --- 1. Config & Setup ---
st.set_page_config(page_title="NS-SUS Defect Inspection", layout="wide")

# ตั้งค่า API Key (ถ้ามี)
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- 2. SIDEBAR CONFIG (แผงควบคุมลับสำหรับคน Demo) ---
st.sidebar.title("🔧 Developer Settings")
use_simulation = st.sidebar.toggle("🎭 Simulation Mode (For Demo)", value=True, help="เปิดโหมดนี้เพื่อจำลองผลลัพธ์โดยไม่ใช้ Quota Google")

if use_simulation:
    st.sidebar.success("✅ SIMULATION ACTIVE: ระบบจะจำลองคำตอบเสมือนจริง (ไม่กิน Quota)")
    force_fail = st.sidebar.checkbox("⚠️ Force Defect (สั่งให้เจอของเสีย)", value=False)
else:
    st.sidebar.warning("📡 LIVE AI MODE: ระบบจะเรียกใช้ Google Gemini จริง (ระวัง Quota)")

# --- 3. LOGIC & DATA (Updated based on NSSUS.pdf) ---
LINE_CONFIG = {
    "CDCM (Continuous Descaling & Cold Rolling)": { 
        # อ้างอิงจาก  รวมกระบวนการ Descaling และ Rolling ไว้ด้วยกัน
        "Product": "CR (Cold Rolled Steel)",
        "Param1": {"name": "Pickling Acid Temp (°C)", "unit": "°C", "default": 85, "min": 60, "max": 100}, # เพิ่มส่วน Descaling 
        "Param2": {"name": "Rolling Force (MN)", "unit": "MN", "default": 1500, "min": 0, "max": 3000},
        "Param3": {"name": "Rolling Speed (mpm)", "unit": "mpm", "default": 1200, "min": 0, "max": 2000},
        "Defect_Focus": "Residual Scale, Pickling stain, Chatter marks, Edge cracks" # เพิ่ม Defect จากการกัดกรด
    },
    "CGL (Continuous Galvanizing Line)": {
        "Product": "GA/GI (Galvanized Steel)",
        "Param1": {"name": "Annealing Furnace Temp (°C)", "unit": "°C", "default": 800, "min": 700, "max": 900}, # เพิ่มส่วน Annealing 
        "Param2": {"name": "Zinc Pot Temp (°C)", "unit": "°C", "default": 460, "min": 440, "max": 480},
        "Param3": {"name": "Air Knife Pressure (kPa)", "unit": "kPa", "default": 40, "min": 0, "max": 100}, # ควบคุม Coating Weight [cite: 80]
        "Defect_Focus": "Dross, Uncoated spots, Zinc adhesion (Peeling), Fluting"
    },
    "EPL (Electrolytic Plating Line)": {
        "Product": "TP/TFS (Tinplate/Tin Free)",
        "Param1": {"name": "Plating Current Density (A/dm²)", "unit": "A/dm²", "default": 20, "min": 0, "max": 100},
        "Param2": {"name": "Reflow Temperature (°C)", "unit": "°C", "default": 250, "min": 230, "max": 300}, # เพิ่ม Reflow Process 
        "Param3": {"name": "Coating Weight (g/m²)", "unit": "g/m²", "default": 2.8, "min": 1.0, "max": 11.0}, # สำคัญสำหรับ TP/TFS 
        "Defect_Focus": "Pinholes, Plating burns (White/Black), Reflow stain, Woodgrain"
    }
}

def save_log(timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level):
    file_name = 'production_logs_v2.csv'
    header_needed = not os.path.isfile(file_name)
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if header_needed:
            writer.writerow(['Timestamp', 'Line', 'Lot No.', 'Param 1', 'Param 2', 'Param 3', 'Status', 'Defect', 'Risk'])
        writer.writerow([timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level])

# --- 4. UI Layout ---
st.title("NS-SUS Defect Inspection")
st.markdown("---")

st.subheader("Select Production Line")
selected_line_name = st.selectbox("Choose Process Unit:", list(LINE_CONFIG.keys()))
current_config = LINE_CONFIG[selected_line_name]

st.markdown(f"**active Module:** `{current_config['Product']}`")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Lot Number**")
        lot_number = st.text_input("Lot No.", value="LOT-2026-X001", label_visibility="collapsed")
    with c2:
        p1_cfg = current_config['Param1']
        st.markdown(f"**{p1_cfg['name']}**")
        p1_val = st.number_input("P1", value=p1_cfg['default'], label_visibility="collapsed")
    with c3:
        p2_cfg = current_config['Param2']
        st.markdown(f"**{p2_cfg['name']}**")
        p2_val = st.number_input("P2", value=p2_cfg['default'], label_visibility="collapsed")
    with c4:
        p3_cfg = current_config['Param3']
        st.markdown(f"**{p3_cfg['name']}**")
        p3_val = st.number_input("P3", value=p3_cfg['default'], label_visibility="collapsed")

st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Visual Inspection (ของจริงอาจใช้ภาพจาก CCTV)")
    uploaded_file = st.file_uploader(f"Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Inspection Point: {selected_line_name}", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Expert Analysis", type="primary", use_container_width=True)

with col_right:
    st.subheader("คำแนะนำจาก AI")
    
    if uploaded_file and run_btn:
        with st.spinner(f"Consulting {selected_line_name} Expert Module..."):
            
            result_text = ""
            status = "PASS"
            
            # === LOGIC การทำงาน ===
            if use_simulation:
                # 🎭 SIMULATION MODE (โหมดการแสดงละคร)
                time.sleep(2.5) # แกล้งรอ 2.5 วินาที ให้ดูสมจริง
                
                if force_fail:
                    # ถ้าสั่งให้ Fail (จำลองการเจอของเสีย)
                    defects = current_config['Defect_Focus'].split(', ')
                    chosen_defect = defects[0] if defects else "Surface Crack"
                    result_text = f"""
                    [STATUS]: FAIL
                    [DEFECT_DETECTED]: {chosen_defect}
                    [CONFIDENCE_SCORE]: 94.5%
                    [ANALYSIS]:
                    - Observation: Detected significant {chosen_defect} on the material surface.
                    - Technical Link: Abnormal parameter settings (P1: {p1_val}) correlated with surface stress.
                    [RECOMMENDED_ACTION]: Immediate stop recommended. Check roller conditions.
                    """
                    status = "FAIL"
                else:
                    # ถ้าให้ Pass (จำลองสินค้าปกติ)
                    result_text = f"""
                    [STATUS]: PASS
                    [DEFECT_DETECTED]: None
                    [CONFIDENCE_SCORE]: 98.2%
                    [ANALYSIS]:
                    - Observation: Surface texture appears consistent and free of defects.
                    - Compliance: Meets strict quality standards for {current_config['Product']}.
                    [RECOMMENDED_ACTION]: Continue production. Parameters are stable.
                    """
                    status = "PASS"
            
            else:
                # 📡 LIVE MODE (ของจริง)
                try:
                    # พยายามเลือก Model
                    try:
                        model = genai.GenerativeModel('gemini-2.5-flash')
                    except:
                        model = genai.GenerativeModel('gemini-pro')

                    prompt = f"""
                    Role: Senior Process Engineer at NS-Siam United Steel. Line: {selected_line_name}.
                    Analyze image for defects: {current_config['Defect_Focus']}.
                    Response: 
                    * [STATUS]: (PASS/FAIL)
                    * [DEFECT_DETECTED]: ...(อธิบายว่าเจออะไร), 
                    * [ANALYSIS]: ...(วิเคราะห์ว่าปัญหาเกิดจากค่าพารามิเตอร์ไหน)
                    * [NEXT STEP]: ...(ต้องแก้ปัญหาจากอะไร)
                    ตอบเป็นภาษาไทย
                    """
                    response = model.generate_content([prompt, image])
                    result_text = response.text
                    # ----------------- แก้ไขตรงนี้ -----------------
                    # เปลี่ยนการเช็กให้ครอบคลุม (เผื่อ AI ตอบ **FAIL** หรือมีเว้นวรรค)
                    if "FAIL" in result_text.upper():
                        status = "FAIL"
                    else:
                        status = "PASS"
                    # -------------------------------------------------------------------
                except Exception as e:
                    st.error(f"⚠️ Live AI Failed (Quota Exceeded?): {e}")
                    st.info("💡 Tip: เปิด 'Simulation Mode' ที่ Sidebar ด้านซ้ายเพื่อ Demo งานต่อได้เลย")
                    status = "ERROR"

            # === DISPLAY RESULT ===
            if status != "ERROR":
                if status == "FAIL":
                    st.error(f"🚨 FAIL: Defect Detected")
                else:
                    st.success(f"✅ PASS: Quality Approved")
                
                with st.container(border=True):
                    st.markdown(result_text)
                    
                # Save Log
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                mode_label = "Simulated" if use_simulation else "AI Check"
                save_log(current_time, selected_line_name, lot_number, p1_val, p2_val, p3_val, status, mode_label, "Low")

st.divider()
st.subheader("History Log")
if os.path.isfile('production_logs_v2.csv'):
    df = pd.read_csv('production_logs_v2.csv')
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
