import streamlit as st
import google.generativeai as genai
from PIL import Image
import csv
import os
import datetime
import pandas as pd

# --- 1. Config & Setup ---
st.set_page_config(page_title="NSSUS Predictive QA", page_icon="🏭", layout="wide")

# ตรวจสอบ API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ ไม่พบ API Key กรุณาตั้งค่าใน Streamlit Secrets ก่อนครับ")
    st.stop()

# --- ฟังก์ชันบันทึก (ใช้ไฟล์ production_logs.csv) ---
def save_log(timestamp, lot_id, machine_temp, pressure, speed, status, prediction, risk_level):
    file_name = 'production_logs.csv'
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Lot No.', 'Temp(C)', 'Pressure(Bar)', 'Speed(m/min)', 'Status', 'AI Prediction', 'Risk Level'])
        writer.writerow([timestamp, lot_id, machine_temp, pressure, speed, status, prediction, risk_level])

# --- 2. UI Setup ---
st.title("🏭 NSSUS Predictive Quality Assurance")
st.caption("ระบบทำนายโอกาสเกิด Defect จากสภาพเครื่องจักรและภาพหน้างาน (CCTV)")

col_control, col_display = st.columns([1, 2])

with col_control:
    st.header("⚙️ Control Panel")
    st.info("ระบุข้อมูลการผลิตปัจจุบัน")
    
    st.markdown("### 📦 Product Identification")
    lot_number = st.text_input("ระบุเลข Lot Number", value="LOT-2026-A001")
    
    st.markdown("---")
    st.markdown("### ⚙️ Machine Parameters")
    
    st.write("🌡️ Temperature (°C) [Normal: 800-900]")
    machine_temp = st.number_input("อุณหภูมิ", min_value=0, max_value=1500, value=850, step=10, label_visibility="collapsed")
    
    st.write("⬇️ Rolling Pressure (Bar)")
    pressure = st.number_input("แรงกด", min_value=0, max_value=1000, value=200, step=5, label_visibility="collapsed")
    
    st.write("⏩ Line Speed (m/min)")
    line_speed = st.number_input("ความเร็วไลน์ผลิต", min_value=0, max_value=3000, value=1200, step=50, label_visibility="collapsed")
    
    st.markdown("---")
    st.header("📹 CCTV Input")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

with col_display:
    st.header("📊 Real-time Monitor")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Monitoring Lot: {lot_number}", width=500)
        
        if st.button("🚀 Run Predictive Analysis", type="primary"):
            if not lot_number:
                st.warning("⚠️ กรุณาระบุเลข Lot Number ก่อนวิเคราะห์ครับ")
            else:
                with st.spinner(f"Analyzing Lot {lot_number}..."):
                    try:
                        # --- 🎯 PROMPT ฉบับแม่นยำ (Calibrated Prompt) ---
                        prompt = f"""
                        Role: You are a Senior QA Engineer at a Steel Factory. 
                        Your job is to prevent FALSE ALARMS. You only flag defects that are clearly visible and affect product quality.
                        
                        Target Product Lot No: {lot_number}
                        
                        Current Machine Conditions:
                        - Temperature: {machine_temp} °C (Normal Range: 800-900)
                        - Pressure: {pressure} Bar
                        - Speed: {line_speed} m/min
                        
                        Standard Acceptance Criteria:
                        1. ACCEPTABLE (Pass): Minor surface texture, water stains, or very faint scratches (light reflection) are NORMAL. Do not flag these.
                        2. REJECT (Fail): Deep cracks, heavy scale, severe scratches, holes, or distinct discoloration.
                        
                        Task:
                        1. Analyzes the image strictly based on the criteria above.
                        2. If the image looks mostly clean or ambiguous -> Result is "PASS".
                        3. If there is a CLEAR defect -> Result is "FAIL".
                        4. Combine visual finding with machine parameters to predict future risk.
                        
                        Response Format (Strictly follow this):
                        [STATUS]: (PASS / FAIL)
                        [DEFECT_TYPE]: (Name of defect OR "None")
                        [ANALYSIS]: (Brief explanation)
                        [RISK_PREDICTION]: (Based on machine params)
                        """
                        
                        # ส่งข้อมูลให้ AI
                        response = model.generate_content([prompt, image])
                        result_text = response.text
                        
                        # --- Logic การแสดงผลแบบใหม่ ---
                        status = "PASS" # Default
                        risk_level = "Low"
                        
                        if "[STATUS]: FAIL" in result_text:
                            status = "FAIL"
                            risk_level = "High"
                            st.error(f"🚨 DETECTED: พบความผิดปกติใน Lot {lot_number}")
                        elif "Critical" in result_text:
                            status = "FAIL"
                            risk_level = "Critical"
                            st.error("🚨 CRITICAL WARNING!")
                        else:
                            status = "PASS"
                            risk_level = "Low"
                            st.success(f"✅ Lot {lot_number} : ผ่านเกณฑ์ (PASS)")
                            
                        st.markdown("### 🧠 AI Analysis Details")
                        st.code(result_text, language='yaml')
                        
                        # บันทึก Log
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # ดึงชื่อ Defect แบบง่ายๆ (ถ้ามี)
                        prediction = "Normal"
                        if "DEFECT_TYPE]:" in result_text:
                            # พยายามตัดคำหลัง : มาแสดง
                            try:
                                prediction = result_text.split("[DEFECT_TYPE]:")[1].split("\n")[0].strip()
                            except:
                                prediction = "See Details"
                        
                        save_log(current_time, lot_number, machine_temp, pressure, line_speed, status, prediction, risk_level)
                        st.toast(f"บันทึกข้อมูล Lot {lot_number} เรียบร้อย!", icon="💾")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Waiting for CCTV Input...")

st.divider()
st.subheader("📜 Production History Log")

log_file = 'production_logs.csv'
if os.path.isfile(log_file):
    try:
        df = pd.read_csv(log_file)
        st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
    except Exception:
        os.remove(log_file)
        st.warning("⚠️ พบไฟล์ Log เสียหาย ระบบทำการรีเซ็ตใหม่แล้ว กรุณากด Run อีกครั้ง")
else:
    st.info("ยังไม่มีข้อมูลในระบบใหม่ (กรุณากด Run Predictive Analysis)")
