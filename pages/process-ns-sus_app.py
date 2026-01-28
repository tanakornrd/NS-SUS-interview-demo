import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. ระบบจัดการฐานข้อมูล (จำลองด้วย CSV)
# ==========================================
DB_FILE = 'tracking_db.csv'

def init_db():
    # ถ้ายังไม่มีไฟล์เก็บข้อมูล ให้สร้างใหม่
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=['Lot_ID', 'Date', 'Complaint', 'Department', 'Status', 'Estimated_Days'])
        df.to_csv(DB_FILE, index=False)

def save_to_db(lot_id, complaint, dept, status, days):
    df = pd.read_csv(DB_FILE)
    new_data = pd.DataFrame({
        'Lot_ID': [lot_id],
        'Date': [datetime.now().strftime("%Y-%m-%d %H:%M")],
        'Complaint': [complaint],
        'Department': [dept],
        'Status': [status],
        'Estimated_Days': [days]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def get_status(lot_id):
    if not os.path.exists(DB_FILE): return None
    df = pd.read_csv(DB_FILE)
    # ค้นหา Lot ID (แบบไม่สนตัวพิมพ์เล็ก-ใหญ่)
    match = df[df['Lot_ID'].astype(str).str.upper() == lot_id.upper()]
    if not match.empty:
        return match.iloc[-1] # เอาข้อมูลล่าสุด
    return None

# เริ่มต้นระบบฐานข้อมูล
init_db()

# ==========================================
# 2. ส่วนสมอง AI (เหมือนเดิม)
# ==========================================
@st.cache_resource
def load_model():
    try:
        df = pd.read_csv('complaints_data.csv') # ต้องมีไฟล์นี้อยู่ข้างนอก pages นะครับ
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(df['text'], df['department'])
        return model
    except Exception as e:
        return None

global_model = load_model()

# ==========================================
# 3. Class SmartClaimTracker (ปรับปรุงให้ Save ได้)
# ==========================================
class SmartClaimTracker:
    def __init__(self, lot_id, complaint):
        self.lot_id = lot_id
        self.complaint = complaint
        self.status = "Received"
        self.department = None
        self.estimated_days = 0
        self.logs = []

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def run_process(self, model, placeholder):
        # Simulation: AI Analysis
        self.log(f"📥 รับข้อมูล Lot No: {self.lot_id}")
        self.log(f"⚠️ อาการเสีย: {self.complaint}")
        placeholder.code("\n".join(self.logs))
        time.sleep(0.5)

        self.log("🤖 AI กำลังวิเคราะห์แยกแผนก...")
        placeholder.code("\n".join(self.logs))
        time.sleep(1.0)

        predicted_dept = model.predict([self.complaint])[0]
        self.department = predicted_dept

        # Logic วันดำเนินการ
        if "R&D" in predicted_dept:
            self.estimated_days = 7
            self.status = "In Analysis Process (R&D)"
        elif "QC" in predicted_dept:
            self.estimated_days = 3
            self.status = "In Lab Testing (QC)"
        else:
            self.estimated_days = 2
            self.status = "Investigating (Logistics)"
            
        self.log(f"✅ Analysis Complete: Forward to {self.department}")
        placeholder.code("\n".join(self.logs))
        
        # Save ลง CSV
        save_to_db(self.lot_id, self.complaint, self.department, self.status, self.estimated_days)

# ==========================================
# 4. หน้าจอหลัก (User Interface)
# ==========================================
st.set_page_config(page_title="Smart Claim Tracking", page_icon="📦")

st.title("📦 NSSUS Smart Claim & Tracking")
st.caption("ระบบรับเรื่องและติดตามสถานะการเคลมสินค้าด้วย AI")

if global_model is None:
    st.error("❌ ไม่พบไฟล์ complaints_data.csv กรุณาเช็คว่าไฟล์อยู่ที่โฟลเดอร์หลักครับ")
else:
    # สร้าง Tab แยกการทำงาน
    tab1, tab2 = st.tabs(["📝 เปิดเคสใหม่ (Submit Case)", "🔍 เช็คสถานะ (Track Status)"])

    # --- TAB 1: สำหรับ TSE คีย์ข้อมูล ---
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Submit New Claim")
            lot_input = st.text_input("ระบุเลข Lot สินค้า (Lot No.):", placeholder="เช่น LOT-2026-001")
            complaint_input = st.text_area("อาการเสีย (Complaint):", height=100, placeholder="เช่น สนิมแดงขึ้นกระจายทั่วแผ่น...")
            
            if st.button("🚀 Process & Save", type="primary"):
                if lot_input and complaint_input:
                    # สร้าง Object และรัน
                    tracker = SmartClaimTracker(lot_input, complaint_input)
                    log_box = st.empty()
                    
                    with st.spinner("AI Processing..."):
                        tracker.run_process(global_model, log_box)
                    
                    st.success(f"✅ บันทึกข้อมูลเรียบร้อย! (Lot: {lot_input})")
                    
                    # สรุปผล
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Department", tracker.department)
                    m2.metric("Status", tracker.status)
                    m3.metric("Est. Time", f"{tracker.estimated_days} Days")
                    
                else:
                    st.warning("กรุณากรอกข้อมูลให้ครบครับ")
        
        with col2:
            st.info("💡 **Tips:**\nเมื่อกดปุ่ม ระบบจะทำการวิเคราะห์ด้วย AI และ **บันทึกลง Database** อัตโนมัติ เพื่อให้ลูกค้าสามารถนำเลข Lot ไปค้นหาได้ใน Tab ถัดไปครับ")

    # --- TAB 2: สำหรับลูกค้าเช็คสถานะ ---
    with tab2:
        st.subheader("🔍 Track Your Claim Status")
        track_id = st.text_input("กรอกเลข Lot ที่ต้องการค้นหา:", placeholder="Enter Lot No...")
        
        if st.button("🔎 Search"):
            if track_id:
                result = get_status(track_id)
                if result is not None:
                    st.success("✅ พบข้อมูลสินค้า")
                    st.markdown(f"### 📦 Lot No: {result['Lot_ID']}")
                    
                    # แสดง Timeline แบบสวยๆ
                    st.progress(60) # สมมติว่า process ไปแล้วระดับนึง
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**📅 วันที่รับเรื่อง:** {result['Date']}")
                        st.write(f"**⚠️ อาการแจ้ง:** {result['Complaint']}")
                    with c2:
                        st.write(f"**📍 แผนกที่ดูแล:** {result['Department']}")
                        st.write(f"**⏱️ เวลาดำเนินการ:** {result['Estimated_Days']} วัน")
                    
                    st.info(f"🚩 **สถานะปัจจุบัน:** {result['Status']}")
                else:
                    st.error(f"❌ ไม่พบข้อมูล Lot Number: {track_id}")
                    st.caption("ลองตรวจสอบเลข Lot อีกครั้ง หรือลองไปสร้างเคสใหม่ที่ Tab แรกครับ")