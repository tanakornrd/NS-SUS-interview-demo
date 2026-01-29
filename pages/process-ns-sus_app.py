import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import io

# ==========================================
# 1. ระบบจัดการฐานข้อมูล (จำลองด้วย CSV)
# ==========================================
DB_FILE = 'tracking_db.csv'

def init_db():
    # กำหนดคอลัมน์ที่ต้องมีทั้งหมด
    expected_columns = ['Lot_ID', 'Date', 'Complaint', 'Department', 'Status', 'Estimated_Days', 'Current_Handler', 'Action_History']
    
    if not os.path.exists(DB_FILE):
        # ถ้าไม่มีไฟล์ สร้างใหม่เลย
        df = pd.DataFrame(columns=expected_columns)
        df.to_csv(DB_FILE, index=False)
    else:
        # 🛠️ AUTO-MIGRATION SYSTEM 🛠️
        # ถ้ามีไฟล์อยู่แล้ว เช็คว่าคอลัมน์ครบไหม ถ้าไม่ครบให้เติม
        df = pd.read_csv(DB_FILE)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        
        if missing_cols:
            # เติมคอลัมน์ที่ขาดด้วยค่า Default
            for col in missing_cols:
                df[col] = "System" if col == 'Current_Handler' else ""
            
            # บันทึกทับไฟล์เดิมทันที
            df.to_csv(DB_FILE, index=False)

def save_to_db(lot_id, complaint, dept, status, days):
    df = pd.read_csv(DB_FILE)
    # ตรวจสอบว่า Lot ID ซ้ำหรือไม่ ถ้าซ้ำให้อัปเดต ถ้าไม่ซ้ำให้เพิ่มใหม่ (ในที่นี้ทำแบบเพิ่มใหม่ง่ายๆ)
    new_data = pd.DataFrame({
        'Lot_ID': [lot_id],
        'Date': [datetime.now().strftime("%Y-%m-%d %H:%M")],
        'Complaint': [complaint],
        'Department': [dept],
        'Status': [status],
        'Estimated_Days': [days],
        'Current_Handler': [dept], # เริ่มต้นให้แผนกที่ AI เลือกเป็นคนถือเรื่อง
        'Action_History': [f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Case Created -> Assigned to {dept}"]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def update_status(lot_id, new_status, action_note, next_handler=None):
    df = pd.read_csv(DB_FILE)
    idx = df[df['Lot_ID'].astype(str) == str(lot_id)].index
    if not idx.empty:
        df.loc[idx, 'Status'] = new_status
        # อัปเดตประวัติการทำงาน
        history = df.loc[idx, 'Action_History'].values[0]
        new_history = f"{history} || [{datetime.now().strftime('%Y-%m-%d %H:%M')}] {action_note}"
        df.loc[idx, 'Action_History'] = new_history
        
        if next_handler:
            df.loc[idx, 'Current_Handler'] = next_handler
            
        df.to_csv(DB_FILE, index=False)
        return True
    return False

def get_all_data():
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    return pd.read_csv(DB_FILE)

# เริ่มต้นระบบฐานข้อมูล
init_db()

# ==========================================
# 2. ส่วนสมอง AI (เหมือนเดิม)
# ==========================================
@st.cache_resource
def load_model():
    try:
        # สร้างไฟล์ตัวอย่างถ้ายังไม่มี เพื่อให้โค้ดรันได้เลยสำหรับการ demo
        if not os.path.exists('complaints_data.csv'):
            data = {
                'text': ['สนิมขึ้น', 'ขนาดไม่ได้', 'ส่งช้า', 'สินค้าบุบ', 'สีเพี้ยน', 'ความแข็งไม่ได้มาตรฐาน', 'ขนส่งทำของเสียหาย'],
                'department': ['QC', 'QC', 'Logistics', 'Logistics', 'QC', 'R&D', 'Logistics']
            }
            pd.DataFrame(data).to_csv('complaints_data.csv', index=False)
            
        df = pd.read_csv('complaints_data.csv')
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(df['text'], df['department'])
        return model
    except Exception as e:
        return None

global_model = load_model()

# ==========================================
# 3. Helper Functions สำหรับ Report
# ==========================================
def generate_report_file(case_data):
    # สร้างเนื้อหา Report
    content = f"""
    ========================================
    NSSUS CLAIM REPORT
    ========================================
    Lot ID: {case_data['Lot_ID']}
    Date: {case_data['Date']}
    Department: {case_data['Department']}
    Current Status: {case_data['Status']}
    
    COMPLAINT DETAIL:
    {case_data['Complaint']}
    
    ACTION HISTORY:
    """
    for action in str(case_data['Action_History']).split(' || '):
        content += f"- {action}\n"
        
    content += f"""
    ========================================
    NEXT STEP RECOMMENDATION:
    - Please verify the resolution with the customer.
    - Archive this case if status is 'Completed'.
    ========================================
    """
    return content

# ==========================================
# 4. หน้าจอหลัก (User Interface)
# ==========================================
st.set_page_config(page_title="Smart Claim Tracking", page_icon="📦", layout="wide")

st.title("📦 NSSUS Smart Claim & Tracking Dashboard")

# สร้าง Tabs หลัก
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard Summary", "📝 Submit New Case", "✅ Workflow Approval", "🔍 Customer Tracking"])

df = get_all_data()

# --- TAB 1: Dashboard Summary ---
with tab1:
    st.header("📈 ภาพรวมการจัดการข้อร้องเรียน")
    
    if not df.empty:
        # Metrics หลัก
        col1, col2, col3, col4 = st.columns(4)
        total_cases = len(df)
        completed_cases = len(df[df['Status'] == 'Completed'])
        pending_cases = total_cases - completed_cases
        
        col1.metric("Total Cases", total_cases)
        col2.metric("Completed", completed_cases)
        col3.metric("Pending", pending_cases)
        col4.metric("Avg. Resolution Time", "2.5 Days") # ตัวอย่าง Mock data
        
        st.divider()
        
        # Charts (ถ้ามีเคส)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("เคสแยกตามแผนก")
            dept_counts = df['Department'].value_counts()
            st.bar_chart(dept_counts)
            
        with c2:
            st.subheader("สถานะงานปัจจุบัน")
            status_counts = df['Status'].value_counts()
            st.bar_chart(status_counts)
            
        st.subheader("📋 รายการเคสล่าสุด")
        st.dataframe(df.tail(10))
    else:
        st.info("ยังไม่มีข้อมูลเคสในระบบ")

# --- TAB 2: Submit New Case (เหมือนเดิม + ปรับปรุง) ---
with tab2:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Submit New Claim")
        lot_input = st.text_input("ระบุเลข Lot สินค้า (Lot No.):", placeholder="เช่น LOT-2026-001")
        complaint_input = st.text_area("อาการเสีย (Complaint):", height=100)
        
        if st.button("🚀 Process & Save", type="primary"):
            if lot_input and complaint_input and global_model:
                with st.spinner("AI Processing..."):
                    time.sleep(1) # Sim delay
                    predicted_dept = global_model.predict([complaint_input])[0]
                    
                    # Status logic
                    status = "Pending Investigation"
                    days = 3
                    if predicted_dept == "R&D":
                        status = "Assigned to R&D"
                        days = 7
                    elif predicted_dept == "QC":
                        status = "Assigned to QC"
                        days = 3
                    else:
                        status = "Assigned to Logistics"
                        days = 2
                        
                    save_to_db(lot_input, complaint_input, predicted_dept, status, days)
                
                st.success(f"✅ บันทึกข้อมูลสำเร็จ! ส่งต่อให้แผนก **{predicted_dept}**")
            else:
                st.warning("กรุณากรอกข้อมูลให้ครบถ้วน")
    
    with col2:
        st.info("💡 **AI Auto-Routing**\nระบบจะวิเคราะห์ข้อความและส่งงานไปยังแผนกที่เกี่ยวข้องอัตโนมัติ (QC, R&D, Logistics) พร้อมตั้ง Status เริ่มต้นให้ทันที")

# --- TAB 3: Workflow Approval (ฟีเจอร์ใหม่) ---
with tab3:
    st.header("✅ Workflow & Action Center")
    st.caption("จำลองหน้าจอสำหรับเจ้าหน้าที่แต่ละฝ่ายเพื่อเข้ามาอัปเดตงาน")
    
    # Filter เลือกแผนก (จำลองการ Login)
    user_dept = st.selectbox("เลือกฝ่ายของคุณ (Simulate User Role):", ["QC", "R&D", "Logistics", "Customer Service", "System Admin"])
    if user_dept == "System Admin":
            # Admin เห็นงานทั้งหมด
            my_tasks = df 
            st.warning("⚠️ คุณกำลังอยู่ในโหมด Admin: เห็นงานทั้งหมดรวมถึงงานที่ยังไม่ระบุฝ่าย")
        else:
            # ฝ่ายอื่นเห็นเฉพาะงานตัวเอง
            my_tasks = df[df['Current_Handler'] == user_dept]
        
        if not my_tasks.empty:
            st.write(f"งานที่รอคุณดำเนินการ ({len(my_tasks)} เคส):")
            # ... (ส่วนแสดงผลเหมือนเดิม) ...
    
    # ดึงงานที่ค้างอยู่ที่ฝ่ายนี้ หรือ งานทั้งหมด
    if not df.empty:
        # Logic กรองงาน: ดูงานที่ Current_Handler ตรงกับ User หรือดูทั้งหมด
        my_tasks = df[df['Current_Handler'] == user_dept]
        
        if not my_tasks.empty:
            st.write(f"งานที่รอคุณดำเนินการ ({len(my_tasks)} เคส):")
            
            for index, row in my_tasks.iterrows():
                with st.expander(f"📌 {row['Lot_ID']} : {row['Complaint'][:50]}..."):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"**อาการ:** {row['Complaint']}")
                        st.markdown(f"**สถานะปัจจุบัน:** `{row['Status']}`")
                        st.markdown(f"**History:**")
                        for h in str(row['Action_History']).split(' || '):
                            st.text(f"- {h}")
                            
                    with c2:
                        st.write("### Action")
                        action_note = st.text_input("บันทึกการแก้ไข:", key=f"note_{row['Lot_ID']}")
                        
                        # ปุ่ม Approve / Forward
                        if st.button("✅ Mark as Fixed / Approve", key=f"btn_{row['Lot_ID']}"):
                            update_status(row['Lot_ID'], "Fixed/Resolved", f"{user_dept}: {action_note}", next_handler="Customer Service")
                            st.success("อัปเดตสถานะเป็น Fixed และส่งต่อให้ Customer Service แล้ว")
                            st.experimental_rerun()
                            
                        # ปุ่ม Download Report
                        report_text = generate_report_file(row)
                        st.download_button(
                            label="📄 Download Report",
                            data=report_text,
                            file_name=f"Report_{row['Lot_ID']}.txt",
                            mime="text/plain",
                            key=f"dl_{row['Lot_ID']}"
                        )
        else:
            st.info(f"ไม่มีงานค้างสำหรับฝ่าย {user_dept} ครับ")
    else:
        st.write("ยังไม่มีข้อมูลในระบบ")

# --- TAB 4: Customer Tracking (เหมือนเดิม) ---
with tab4:
    st.subheader("🔍 Track Your Claim Status")
    track_id = st.text_input("กรอกเลข Lot ที่ต้องการค้นหา:", placeholder="Enter Lot No...", key="track_input")
    
    if st.button("🔎 Search", key="track_btn"):
        # อ่านไฟล์ใหม่เสมอเพื่อให้ได้ข้อมูลล่าสุด
        df_latest = get_all_data()
        if not df_latest.empty:
            result = df_latest[df_latest['Lot_ID'].astype(str) == str(track_id)]
            
            if not result.empty:
                res = result.iloc[-1]
                st.success("✅ พบข้อมูลสินค้า")
                
                # Progress Bar ตามสถานะ (Mock logic)
                status_val = 20
                if "Assigned" in res['Status']: status_val = 40
                if "Fixed" in res['Status']: status_val = 80
                if "Completed" in res['Status']: status_val = 100
                st.progress(status_val)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Lot ID:** {res['Lot_ID']}")
                    st.markdown(f"**Status:** `{res['Status']}`")
                with c2:
                    st.markdown(f"**Department:** {res['Department']}")
                    st.markdown(f"**Handler:** {res['Current_Handler']}")
                
                with st.expander("ดูประวัติการดำเนินการ (Timeline)"):
                    for h in str(res['Action_History']).split(' || '):
                        st.write(f"• {h}")
            else:
                st.error("ไม่พบข้อมูล Lot ID นี้")
