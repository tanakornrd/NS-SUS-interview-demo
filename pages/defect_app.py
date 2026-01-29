import streamlit as st
import os
import sys
import subprocess
import time

# --- 0. FORCE UPDATE SYSTEM (ระบบบังคับอัปเกรดสมอง AI) ---
# ส่วนนี้จะทำงานก่อน Code อื่นเพื่อแก้ปัญหา Server ดื้อไม่ยอมอัปเดต
try:
    import google.generativeai as genai
    # เช็กเวอร์ชันปัจจุบัน
    current_ver = genai.__version__
    st.toast(f"System Check: AI Library Version {current_ver}", icon="ℹ️")
    
    # ถ้าเวอร์ชันต่ำกว่า 0.8.3 หรือเป็นตัว Beta เก่าๆ ให้สั่งลงใหม่เดี๋ยวนี้
    if current_ver < "0.8.3":
        st.warning(f"⚠️ Found old library ({current_ver}). Forcing upgrade...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "google-generativeai>=0.8.3"])
        st.rerun() # รีสตาร์ทแอปทันทีหลังจากลงเสร็จ
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai>=0.8.3"])
    st.rerun()
except Exception as e:
    # ถ้าลงไม่ได้จริงๆ ให้ปล่อยผ่านไปก่อน
    pass

# --- เริ่มต้น Import ปกติ ---
import google.generativeai as genai
from PIL import Image
import csv
import pandas as pd

# --- 1. Config & Setup ---
st.set_page_config(page_title="NSSUS Universal QA", page_icon="🏭", layout="wide")

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    
    # 🧠 SMART MODEL SELECTOR (ระบบเลือกโมเดลอัจฉริยะ)
    # พยายามใช้ 1.5 Flash ก่อน ถ้าไม่ได้จะถอยไปใช้ตัวอื่นที่ Server รู้จัก
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Test Call
        model.generate_content("test")
        st.toast("✅ Connected to: Gemini 1.5 Flash", icon="🚀")
    except:
        try:
            # ถ้า 1.5 Flash พัง ให้ลองรุ่น Latest
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            model.generate_content("test")
            st.toast("✅ Connected to: Gemini 1.5 Flash (Latest)", icon="🚀")
        except:
            # ถ้ายังไม่ได้ ให้ใช้ 'gemini-pro-vision' (รุ่นเก่าที่รองรับรูปภาพแน่นอน)
            # เพื่อให้ Demo ไม่ล่มกลางคัน
            model = genai.GenerativeModel('gemini-pro-vision')
            st.toast("⚠️ Fallback Mode: Gemini Pro Vision", icon="🛡️")

else:
    st.error("❌ ไม่พบ API Key กรุณาตรวจสอบใน Secrets")
    st.stop()

# --- 2. KNOWLEDGE BASE (ข้อมูลไลน์ผลิต) ---
LINE_CONFIG = {
    "CDCM (Cold Rolling Mill)": {
        "Product": "CR (Cold Rolled Steel)",
        "Param1": {"name": "Rolling Force", "unit": "MN", "default": 1500, "min": 0, "max": 3000},
        "Param2": {"name": "Strip Tension", "unit": "kN", "default": 50, "min": 0, "max": 200},
        "Param3": {"name": "Rolling Speed", "unit": "mpm", "default": 1200, "min": 0, "max": 2000},
        "Defect_Focus": "Scale, Chatter marks, Edge cracks, Shape defects (Buckle)"
    },
    "CGL (Continuous Galvanizing Line)": {
        "Product": "GA/GI (Galvanized Steel)",
        "Param1": {"name": "Zinc Pot Temp", "unit": "°C", "default": 460, "min": 400, "max": 500},
        "Param2": {"name": "Air Knife Pressure", "unit": "kPa", "default": 40, "min": 0, "max": 100},
        "Param3": {"name": "Line Speed", "unit": "mpm", "default": 180, "min": 0, "max": 300},
        "Defect_Focus": "Dross, Spangle defects, Uncoated spots, Zinc adhesion issues"
    }
}

# --- 3. Save Function ---
def save_log(timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level):
    file_name = 'production_logs_v2.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Line', 'Lot No.', 'Param 1', 'Param 2', 'Param 3', 'Status', 'Defect', 'Risk'])
        writer.writerow([timestamp, line_name, lot_id, p1_val, p2_val, p3_val, status, defect_type, risk_level])

# --- 4. UI Layout ---
st.title("🏭 NSSUS Universal Process QA")
st.markdown("---")

# Select Line
st.subheader("📍 Select Production Line")
selected_line_name = st.selectbox("Choose Process Unit:", list(LINE_CONFIG.keys()))
current_config = LINE_CONFIG[selected_line_name]

st.markdown(f"**active Module:** `{current_config['Product']}`")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**📦 Lot Number**")
        lot_number = st.text_input("Lot No.", value="LOT-2026-X001", label_visibility="collapsed")
    with c2:
        p1_cfg = current_config['Param1']
        st.markdown(f"**⚙️ {p1_cfg['name']}**")
        p1_val = st.number_input("P1", value=p1_cfg['default'], label_visibility="collapsed")
    with c3:
        p2_cfg = current_config['Param2']
        st.markdown(f"**⚙️ {p2_cfg['name']}**")
        p2_val = st.number_input("P2", value=p2_cfg['default'], label_visibility="collapsed")
    with c4:
        p3_cfg = current_config['Param3']
        st.markdown(f"**⏩ {p3_cfg['name']}**")
        p3_val = st.number_input("P3", value=p3_cfg['default'], label_visibility="collapsed")

st.markdown("---")

# Analysis Section
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1️⃣ Visual Inspection")
    uploaded_file = st.file_uploader(f"Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Inspection Point: {selected_line_name}", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Expert Analysis", type="primary", use_container_width=True)

with col_right:
    st.subheader("2️⃣ AI Expert Findings")
    
    if uploaded_file and run_btn:
        with st.spinner(f"Consulting {selected_line_name} Expert Module..."):
            try:
                # Prompt setup
                prompt = f"""
                Role: Senior Process Engineer at NSSUS. Line: {selected_line_name}.
                Analyze image for defects: {current_config['Defect_Focus']}.
                Machine Data: {p1_val}, {p2_val}, {p3_val}.
                Task: Detect defects and decide Pass/Fail.
                Response:
                [STATUS]: (PASS / FAIL)
                [DEFECT_DETECTED]: (Name)
                [ANALYSIS]: (Explanation)
                """
                
                # เรียก AI (ใช้ Model ที่เลือกมาแล้วข้างบน)
                response = model.generate_content([prompt, image])
                result_text = response.text
                
                if "[STATUS]: FAIL" in result_text or "Critical" in result_text:
                    st.error(f"🚨 FAIL: Defect Detected")
                    status = "FAIL"
                else:
                    st.success(f"✅ PASS: Quality Approved")
                    status = "PASS"
                
                with st.container(border=True):
                    st.markdown(result_text)
                    
                # Save Log
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_log(current_time, selected_line_name, lot_number, p1_val, p2_val, p3_val, status, "AI Check", "Low")
                
            except Exception as e:
                st.error(f"Processing Error: {e}")
