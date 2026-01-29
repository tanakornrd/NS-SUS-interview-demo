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
# 1. ระบบจัดการฐานข้อมูล (พร้อมระบบ Auto-Fix)
# ==========================================
DB_FILE = 'tracking_db.csv'

def init_db():
    expected_columns = ['Lot_ID', 'Date', 'Complaint', 'Department', 'Status', 'Estimated_Days', 'Current_Handler', 'Action_History', 'Final_Decision', 'Resolution_Note']
    
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=expected_columns)
        df.to_csv(DB_FILE, index=False)
    else:
        # 🛠️ AUTO-MIGRATION & FIXER SYSTEM 🛠️
        df = pd.read_csv(DB_FILE)
        
        # 1. เติมคอลัมน์ที่ขาด
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                if col == 'Current_Handler': df[col] = "System"
                elif col == 'Status': df[col] = "Pending"
                else: df[col] = ""
        
        # 2. 🚑 DATA FIXER: ซ่อมงานที่ Handler เป็น 'System' ให้ตรงกับ Department
        # ถ้าคนถือเรื่องเป็น "System" หรือว่างเปล่า -> ให้โอนงานไปที่ Department หลักทันที
        if 'Current_Handler' in df.columns and 'Department' in df.columns:
            mask = (df['Current_Handler'] == "System") | (df['Current_Handler'].isna()) | (df['Current_Handler'] == "")
            if mask.any():
                df.loc[mask, 'Current_Handler'] = df.loc[mask, 'Department']
        
        df.to_csv(DB_FILE, index=False)

def save_to_db(lot_id, complaint, dept, status, days):
    df = pd.read_csv(DB_FILE)
    new_data = pd.DataFrame({
        'Lot_ID': [lot_id],
        'Date': [datetime.now().strftime("%Y-%m-%d %H:%M")],
        'Complaint': [complaint],
        'Department': [dept],
        'Status': [status],
        'Estimated_Days': [days],
        'Current_Handler': [dept], # บังคับให้คนรับผิดชอบคนแรก คือแผนกนั้นๆ เสมอ
        'Action_History': [f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Case Created -> Assigned to {dept}"],
        'Final_Decision': [""],
        'Resolution_Note': [""]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def update_status(lot_id, new_status, action_note, next_handler=None, final_decision=None, resolution_note=None):
    df = pd.read_csv(DB_FILE)
    idx = df[df['Lot_ID'].astype(str) == str(lot_id)].index
    if not idx.empty:
        df.loc[idx, 'Status'] = new_status
        history = df.loc[idx, 'Action_History'].values[0]
        new_history = f"{history} || [{datetime.now().strftime('%Y-%m-%d %H:%M')}] {action_note}"
        df.loc[idx, 'Action_History'] = new_history
        
        if next_handler:
            df.loc[idx, 'Current_Handler'] = next_handler
        if final_decision:
             df.loc[idx, 'Final_Decision'] = final_decision
        if resolution_note:
             df.loc[idx, 'Resolution_Note'] = resolution_note
            
        df.to_csv(DB_FILE, index=False)
        return True
    return False

def get_all_data():
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    return pd.read_csv(DB_FILE)

# เรียกใช้ฟังก์ชันซ่อมฐานข้อมูลทันทีที่รัน
init_db()

# ==========================================
# 2. ส่วนสมอง AI
# ==========================================
@st.cache_resource
def load_model():
    try:
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
# 3. User Interface
# ==========================================
st.set_page_config(page_title="Smart Claim Tracking", page_icon="📦", layout="wide")

# --- SIDEBAR (เครื่องมือ Admin) ---
with st.sidebar:
    st.title("🔧 Admin Tools")
    st.info("ใช้ปุ่มนี้หากต้องการล้างข้อมูลเก่าทิ้งเพื่อเริ่ม Demo ใหม่")
    if st.button("🗑️ Reset Database (Clear All)", type="primary"):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            init_db() # สร้างไฟล์ใหม่ทันที
            st.success("Database cleared! Please refresh the page.")
            time.sleep(1)
            st.rerun()

st.title("📦 NSSUS Smart Claim & Tracking Center")

tab1, tab2, tab3 = st.tabs(["📝 Submit & History Log", "✅ Workflow Approval", "🔍 Customer Tracking"])

df = get_all_data()

# --- TAB 1: Submit & History ---
with tab1:
    with st.container():
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("➕ Create New Case")
            col_in1, col_in2 = st.columns([1, 2])
            with col_in1:
                lot_input = st.text_input("Lot No.", placeholder="e.g., LOT-2026-001")
            with col_in2:
                complaint_input = st.text_input("Issue / Complaint", placeholder="Describe the defect...")
            
            if st.button("🚀 Process & Save", type="primary", use_container_width=True):
                if lot_input and complaint_input and global_model:
                    with st.spinner("AI Categorizing..."):
                        time.sleep(0.5)
                        predicted_dept = global_model.predict([complaint_input])[0]
                        status = f"Assigned to {predicted_dept}"
                        days = 3
                        if predicted_dept == "R&D": days = 7
                        elif predicted_dept == "Logistics": days = 2
                        save_to_db(lot_input, complaint_input, predicted_dept, status, days)
                    st.success(f"New case created! Assigned to **{predicted_dept}**")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.warning("Please fill in all fields.")
        
        with c2:
            if not df.empty:
                total = len(df)
                pending = len(df[df['Status'] != 'Case Closed'])
                st.metric("Total Cases", total)
                st.metric("Pending Action", pending, delta_color="inverse")
            else:
                st.info("No Data")

    st.divider()
    st.subheader("📜 Case History Log")
    if not df.empty:
        df_display = df.iloc[::-1].copy()
        df_display = df_display[['Lot_ID', 'Date', 'Complaint', 'Department', 'Status', 'Current_Handler']]
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Lot_ID": st.column_config.TextColumn("Lot Number", width="medium"),
                "Date": st.column_config.TextColumn("Timestamp", width="small"),
                "Complaint": st.column_config.TextColumn("Issue Description", width="large"),
                "Department": st.column_config.TextColumn("Dept", width="small"),
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "Current_Handler": st.column_config.TextColumn("Handler (Who has the ball?)", width="medium"),
            }
        )
    else:
        st.info("No history data available.")

# --- TAB 2: Workflow ---
with tab2:
    st.header("✅ Workflow & Action Center")
    # ตัวเลือก User Role
    user_dept = st.selectbox("Select User Role (Login As):", ["QC", "R&D", "Logistics", "Customer Service", "System Admin"])
    
    subtab_active, subtab_history = st.tabs(["⚡ Pending Tasks", "📜 Completed History"])

    if not df.empty:
        completed_tasks = df[df['Status'] == 'Case Closed']
        active_tasks_all = df[df['Status'] != 'Case Closed']
        
        my_active_tasks = pd.DataFrame()
        if user_dept == "System Admin":
            my_active_tasks = active_tasks_all
            st.caption("ℹ️ Admin Mode: Viewing all active tasks.")
        else:
            # กรองงานตามแผนกของผู้ใช้
            if 'Current_Handler' in df.columns:
                my_active_tasks = active_tasks_all[active_tasks_all['Current_Handler'] == user_dept]

        with subtab_active:
            if not my_active_tasks.empty:
                for index, row in my_active_tasks.iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown(f"#### 📌 {row['Lot_ID']}")
                            st.markdown(f"**Issue:** {row['Complaint']}")
                            st.caption(f"Status: {row['Status']} | Handler: {row['Current_Handler']}")
                            with st.expander("History Log"):
                                if pd.notna(row['Action_History']):
                                    for h in str(row['Action_History']).split(' || '):
                                        st.write(f"- {h}")

                        with c2:
                            if row['Current_Handler'] == "Customer Service":
                                decision = st.selectbox("Decision", ["Approve", "Compromise", "Reject"], key=f"d_{row['Lot_ID']}")
                                note = st.text_input("Note to Customer", key=f"n_{row['Lot_ID']}")
                                if st.button("🏁 Close Case", key=f"btn_{row['Lot_ID']}", type="primary"):
                                    update_status(row['Lot_ID'], "Case Closed", f"CS: {decision}", "Completed", decision, note)
                                    st.rerun()
                            else:
                                note = st.text_input("Investigation Note", key=f"in_{row['Lot_ID']}")
                                if st.button("➡️ Forward to CS", key=f"fwd_{row['Lot_ID']}"):
                                    update_status(row['Lot_ID'], "Investigation Complete", f"{user_dept}: {note}", "Customer Service")
                                    st.rerun()
            else:
                st.success(f"🎉 No pending tasks for **{user_dept}**")
                st.caption(f"Check if the task is assigned to another department or System Admin.")

        with subtab_history:
            st.dataframe(completed_tasks, use_container_width=True)

# --- TAB 3: Tracking ---
with tab3:
    st.subheader("🔍 Customer Status Check")
    track_id = st.text_input("Enter Lot No.", placeholder="LOT-XXXX-XXX")
    if st.button("Search"):
        df_latest = get_all_data()
        if not df_latest.empty:
            res = df_latest[df_latest['Lot_ID'].astype(str) == str(track_id)]
            if not res.empty:
                r = res.iloc[-1]
                st.success("✅ Found Case")
                st.progress(100 if r['Status'] == 'Case Closed' else 50)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Lot ID:** {r['Lot_ID']}")
                    st.write(f"**Status:** {r['Status']}")
                with c2:
                    st.write(f"**Dept:** {r['Department']}")
                    st.write(f"**Handler:** {r['Current_Handler']}")
                
                if r['Status'] == 'Case Closed':
                    st.divider()
                    st.info(f"**Final Decision:** {r['Final_Decision']}\n\n**Note:** {r['Resolution_Note']}")
            else:
                st.error("Not Found")
