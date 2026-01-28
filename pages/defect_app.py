import streamlit as st
import google.generativeai as genai
from PIL import Image
import csv
import os
import datetime
import pandas as pd

# --- 1. Config & Setup (แก้เรื่อง API Key และ Model ให้ชัวร์) ---
st.set_page_config(page_title="NSSUS Predictive QA", page_icon="🏭", layout="wide")

# ตรวจสอบ API Key ใน Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    # ✅ สร้าง model เตรียมไว้เลย (แก้ปัญหา model is not defined)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("❌ ไม่พบ API Key กรุณาตั้งค่าใน Streamlit Secrets ก่อนครับ")
    st.stop() # หยุดการทำงานถ้ารหัสไม่ครบ

# --- ฟังก์ชันบันทึก ---
def save_log(timestamp, machine_temp, pressure, speed, prediction, risk_level):
    file_name = 'defect_history.csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Temp(C)', 'Pressure(Bar)', 'Speed(m/min)', 'AI Prediction', 'Risk Level'])
        writer.writerow([timestamp, machine_temp, pressure, speed, prediction, risk_level])

# --- 2. UI Setup ---
st.title("🏭 NSSUS Predictive Quality Assurance")
st.caption("ระบบทำนายโอกาสเกิด Defect จากสภาพเครื่องจักรและภาพหน้างาน (CCTV)")

# แบ่งหน้าจอเป็น ซ้าย (Control) : ขวา (Display)
col_control, col_display = st.columns([1, 2])

with col_control:
    st.header("⚙️ Machine Conditions")
    st.info("ระบุค่า Parameter ของเครื่องจักร")
    
    # ✅ อัปเกรด: ใช้ number_input แทน slider เพื่อให้กรอกเลขได้เป๊ะๆ
    # (แต่ยังกด +/- ได้เหมือน Slider)
    
    st.markdown("---")
    st.write("🌡️ Temperature (°C)")
    machine_temp = st.number_input("อุณหภูมิ", min_value=0, max_value=1500, value=850, step=10, label_visibility="collapsed")
    
    st.write("⬇️ Rolling Pressure (Bar)")
    pressure = st.number_input("แรงกด", min_value=0, max_value=1000, value=200, step=5, label_visibility="collapsed")
    
    st.write("⏩ Line Speed (m/min)")
    line_speed = st.number_input("ความเร็วไลน์ผลิต", min_value=0, max_value=3000, value=1200, step=50, label_visibility="collapsed")
    st.markdown("---")
    
    st.header("📹 CCTV Feed Input")
    uploaded_file = st.file_uploader("Upload Image from Camera", type=["jpg", "png", "jpeg"])

with col_display:
    st.header("📊 Real-time Analysis Monitor")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Current Frame: Rolling Stand No.2", width=500)
        
        # ปุ่มกดเพื่อจำลองการ Trigger
        if st.button("🚀 Run Predictive Analysis", type="primary"):
            with st.spinner("Processing Sensor Data & Image..."):
                try:
                    # --- หัวใจสำคัญ: Prompt แบบ Predictive ---
                    prompt = f"""
                    Context: You are a QA Engineer at a Steel Factory.
                    
                    Current Machine Conditions:
                    - Temperature: {machine_temp} °C
                    - Rolling Pressure: {pressure} Bar
                    - Line Speed: {line_speed} m/min
                    
                    Task: 
                    1. Analyze the attached image for any visual anomalies.
                    2. Combine visual findings with the machine conditions above.
                    3. PREDICT what defect is likely to occur if the machine continues running at these settings.
                    
                    Response Format:
                    [RISK_LEVEL]: (Low / Medium / High / Critical)
                    [PREDICTION]: (Name of potential defect, e.g., Scale, Edge Crack)
                    [ADVICE]: (Immediate action required for the operator)
                    ตอบเป็นภาษาไทย
                    """
                    
                    # ส่งรูปและ prompt เข้าโมเดล
                    response = model.generate_content([prompt, image])
                    result_text = response.text
                    
                    # Logic การแสดงผลตามความเสี่ยง
                    if "High" in result_text or "Critical" in result_text:
                        st.error("🚨 WARNING: High Defect Probability Detected!")
                    elif "Medium" in result_text:
                        st.warning("⚠️ Caution: Abnormal Condition Warning")
                    else:
                        st.success("✅ System Normal: Optimal Conditions")
                        
                    # แสดงผลการวิเคราะห์
                    st.markdown("### 🧠 AI Assessment")
                    st.write(result_text)
                    
                    # บันทึก Log
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # ดึง Risk Level แบบง่ายๆ
                    risk_level = "Low"
                    if "Critical" in result_text: risk_level = "Critical"
                    elif "High" in result_text: risk_level = "High"
                    elif "Medium" in result_text: risk_level = "Medium"
                    
                    save_log(current_time, machine_temp, pressure, line_speed, result_text, risk_level)
                    st.toast("บันทึกข้อมูลเรียบร้อยแล้ว", icon="💾")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Waiting for CCTV Input... (Please upload an image)")

# --- ส่วนแสดง History ด้านล่าง ---
st.divider()
st.subheader("📜 Detection Log History")
if os.path.isfile('defect_history.csv'):
    df = pd.read_csv('defect_history.csv')
    # แสดงตารางแบบเรียงจากใหม่สุดไปเก่าสุด
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
