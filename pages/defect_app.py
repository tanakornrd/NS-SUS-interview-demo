import streamlit as st
import google.generativeai as genai
from PIL import Image
import csv
import os
import datetime
import pandas as pd

# --- 1. Config & Setup ---
# ⚠️ สำคัญ: ใน Code จริงควรซ่อน API Key ไม่ให้ใครเห็นครับ (ใช้ st.secrets)
GOOGLE_API_KEY = "AIzaSyBCPSibe8SD3TnEJe0IXw3RDvWi9nTshOo" 
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

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
st.set_page_config(page_title="NSSUS Predictive QA", page_icon="🏭", layout="wide")

st.title("🏭 NSSUS Predictive Quality Assurance")
st.caption("ระบบทำนายโอกาสเกิด Defect จากสภาพเครื่องจักรและภาพหน้างาน (CCTV)")

# แบ่งหน้าจอเป็น ซ้าย (Control) : ขวา (Display)
col_control, col_display = st.columns([1, 2])

with col_control:
    st.header("⚙️ Machine Conditions")
    st.info("จำลองข้อมูลจาก Sensors ในไลน์ผลิต")
    
    # Simulation Sliders
    machine_temp = st.slider("🌡️ Temperature (°C)", 0, 1000, 850)
    pressure = st.slider("⬇️ Rolling Pressure (Bar)", 0, 500, 200)
    line_speed = st.slider("⏩ Line Speed (m/min)", 0, 2000, 1200)
    
    st.divider()
    
    st.header("📹 CCTV Feed Input")
    uploaded_file = st.file_uploader("Image from Camera 01", type=["jpg", "png", "jpeg"])

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
                    # เราส่งทั้ง "ค่าตัวเลข" และ "รูปภาพ" ให้ AI ประมวลผลร่วมกัน
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
                    """
                    
                    response = model.generate_content([prompt, image])
                    result_text = response.text
                    
                    # Logic การแสดงผลตามความเสี่ยง
                    if "High" in result_result_text or "Critical" in result_text:
                        st.error("🚨 WARNING: High Defect Probability Detected!")
                        st.audio("https://upload.wikimedia.org/wikipedia/commons/d/d1/Car_Horn.wav") # เสียงเตือนจำลอง
                    elif "Medium" in result_text:
                        st.warning("⚠️ Caution: Abnormal Condition Warning")
                    else:
                        st.success("✅ System Normal: Optimal Conditions")
                        
                    # แสดงผลการวิเคราะห์
                    st.markdown("### 🧠 AI Assessment")
                    st.write(result_text)
                    
                    # บันทึก Log
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # ดึง Risk Level แบบง่ายๆ (ตัดคำ)
                    risk_level = "Unknown"
                    if "Critical" in result_text: risk_level = "Critical"
                    elif "High" in result_text: risk_level = "High"
                    elif "Medium" in result_text: risk_level = "Medium"
                    else: risk_level = "Low"
                    
                    save_log(current_time, machine_temp, pressure, line_speed, result_text, risk_level)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Waiting for CCTV Input... (Please upload an image)")

# --- ส่วนแสดง History ด้านล่าง ---
st.divider()
st.subheader("📜 Detection Log History")
if os.path.isfile('defect_history.csv'):
    df = pd.read_csv('defect_history.csv')
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
