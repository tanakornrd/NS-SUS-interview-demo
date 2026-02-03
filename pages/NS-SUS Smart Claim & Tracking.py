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
# 1. ระบบจัดการฐานข้อมูล
# ==========================================
DB_FILE = 'tracking_db_v3_mcs.csv' 

def init_db():
    expected_columns = ['Lot_ID', 'Date', 'Complaint', 'Department', 'Status', 'Estimated_Days', 'Current_Handler', 'Action_History', 'Final_Decision', 'Resolution_Note']
    
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=expected_columns)
        df.to_csv(DB_FILE, index=False)
    else:
        df = pd.read_csv(DB_FILE)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            for col in missing_cols: df[col] = ""
        
        if 'Current_Handler' in df.columns and 'Department' in df.columns:
            mask = (df['Current_Handler'] == "System") | (df['Current_Handler'].isnull())
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
        'Current_Handler': [dept], 
        'Action_History': [f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Case Created -> Assigned to {dept}"],
        'Final_Decision': [""],
        'Resolution_Note': [""]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def update_status(lot_id, new_status, action_note, next_handler=None, final_decision=None, resolution_note=None, force_handler=None):
    df = pd.read_csv(DB_FILE)
    idx = df[df['Lot_ID'].astype(str) == str(lot_id)].index
    if not idx.empty:
        if force_handler:
            df.loc[idx, 'Current_Handler'] = force_handler
            if "Case Closed" not in new_status: 
                 df.loc[idx, 'Department'] = force_handler
            action_note += f" (Management re-assigned to {force_handler})"

        df.loc[idx, 'Status'] = new_status
        history = df.loc[idx, 'Action_History'].values[0]
        new_history = f"{history} || [{datetime.now().strftime('%Y-%m-%d %H:%M')}] {action_note}"
        df.loc[idx, 'Action_History'] = new_history
        
        if next_handler: df.loc[idx, 'Current_Handler'] = next_handler
        if final_decision: df.loc[idx, 'Final_Decision'] = final_decision
        if resolution_note: df.loc[idx, 'Resolution_Note'] = resolution_note
            
        df.to_csv(DB_FILE, index=False)
        return True
    return False

def get_all_data():
    if not os.path.exists(DB_FILE): return pd.DataFrame()
    return pd.read_csv(DB_FILE)

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
st.set_page_config(page_title="NS-SUS Smart Claim & Tracking", layout="wide")

with st.sidebar:
    st.title("🔧 Tools")
    if st.button("🗑️ Reset Database (Clear All)", type="primary"):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            init_db()
            st.success("Database Cleared! 🧹")
            time.sleep(1)
            st.rerun()

st.title("NSSUS Smart Claim Tracking")

# ปรับ Layout เป็น 4 Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Executive Dashboard", "Submit & Log", "Workflow", "Customer Tracking"])

df = get_all_data()

# --- TAB 1: EXECUTIVE DASHBOARD (แทน Power BI) ---
with tab1:
    st.markdown("###Real-time Analytics Dashboard")
    st.caption("ข้อมูลวิเคราะห์สถานการณ์เคลมสินค้าแบบ Real-time (ทดแทน Power BI)")
    
    if not df.empty:
        # 1. KPI Cards (ตัวเลขสำคัญ)
        col1, col2, col3, col4 = st.columns(4)
        total = len(df)
        closed = len(df[df['Status'] == 'Case Closed'])
        active = total - closed
        # คำนวณ % การปิดงาน
        success_rate = (closed / total) * 100 if total > 0 else 0
        
        col1.metric("Total Claims", total, help="จำนวนเคสทั้งหมด")
        col2.metric("Resolved", closed, delta=f"{success_rate:.1f}% Rate")
        col3.metric("Active Issues", active, delta_color="inverse")
        col4.metric("Avg. Resolution", "2.1 Days") # (Mockup)
        
        st.divider()
        
        # 2. Charts Zone
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("ปัญหาแยกตามแผนก (Defects by Dept)")
            if 'Department' in df.columns:
                dept_counts = df['Department'].value_counts()
                st.bar_chart(dept_counts, color="#FF4B4B") # สีแดง NSSUS
        
        with c2:
            st.subheader("สถานะงาน (Work Status)")
            if 'Status' in df.columns:
                # ทำข้อมูลให้เป็นกลุ่มๆ เพื่อกราฟสวย
                status_counts = df['Status'].value_counts()
                st.bar_chart(status_counts, color="#29B5E8") # สีฟ้า
                
        # 3. Business Outcome (ผลประกอบการ)
        st.subheader("ผลการพิจารณา")
        if 'Final_Decision' in df.columns:
            outcomes = df[df['Final_Decision'] != ""]['Final_Decision'].value_counts()
            if not outcomes.empty:
                st.bar_chart(outcomes, horizontal=True) # กราฟแนวนอน
            else:
                st.info("ยังไม่มีข้อมูลการตัดสินใจปิดเคส (No finalized cases)")
                
    else:
        st.info("Waiting for data stream... Please submit a case in the next tab.")

# --- TAB 2: Submit & Log ---
with tab2:
    with st.container():
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("➕ Create New Case")
            col_in1, col_in2 = st.columns([1, 2])
            with col_in1:
                lot_input = st.text_input("Lot No.", placeholder="e.g., LOT-2026-001")
            with col_in2:
                complaint_input = st.text_input("Issue / Complaint", placeholder="Describe the defect...")
            
            if st.button("Process & Save", type="primary", use_container_width=True):
                if lot_input and complaint_input and global_model:
                    with st.spinner("AI Categorizing..."):
                        time.sleep(0.5)
                        predicted_dept = global_model.predict([complaint_input])[0]
                        status = f"Assigned to {predicted_dept}"
                        days = 3
                        if predicted_dept == "R&D": days = 7
                        elif predicted_dept == "Logistics": days = 2
                        save_to_db(lot_input, complaint_input, predicted_dept, status, days)
                    st.success(f"New case assigned to **{predicted_dept}**")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.warning("Please fill in all fields.")
        
        with c2:
             # ปุ่ม Excel Export
            st.write("### 📥 Export Data")
            if not df.empty:
                buffer = io.BytesIO()
                # เปลี่ยนเป็น CSV เพื่อความชัวร์บน Mac (ไม่ต้องลง lib เพิ่ม)
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV Report",
                    data=csv_data,
                    file_name="NSSUS_Report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.divider()
    st.subheader("Operational Log")
    if not df.empty:
        df_display = df.iloc[::-1].copy()
        st.dataframe(
            df_display[['Lot_ID', 'Date', 'Complaint', 'Department', 'Status', 'Current_Handler']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No data available.")

# --- TAB 3: Workflow ---
with tab3:
    st.header("Workflow & Action Center")
    user_roles = ["QC", "R&D", "QA", "Marketing & Customer Service (MCS)"]
    user_dept = st.selectbox("Login As:", user_roles)
    
    subtab_active, subtab_history = st.tabs(["Pending Tasks", "Completed History"])

    if not df.empty:
        completed_tasks = df[df['Status'] == 'Case Closed']
        active_tasks_all = df[df['Status'] != 'Case Closed']
        
        my_active_tasks = pd.DataFrame()
        if user_dept == "Marketing & Customer Service (MCS)":
            my_active_tasks = active_tasks_all
            st.success("MCS Mode")
        else:
            if 'Current_Handler' in df.columns:
                my_active_tasks = active_tasks_all[active_tasks_all['Current_Handler'] == user_dept]

        with subtab_active:
            if not my_active_tasks.empty:
                for index, row in my_active_tasks.iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([1.5, 1])
                        with c1:
                            st.markdown(f"#### 📌 {row['Lot_ID']}")
                            st.markdown(f"**Issue:** {row['Complaint']}")
                            st.info(f"**Current Handler:** {row['Current_Handler']}") 
                            with st.expander("History Log"):
                                if pd.notna(row['Action_History']):
                                    for h in str(row['Action_History']).split(' || '):
                                        st.caption(f"• {h}")
                        with c2:
                            st.write("### Action")
                            if user_dept == "Marketing & Customer Service (MCS)":
                                if row['Current_Handler'] == "Marketing & Customer Service (MCS)":
                                    st.markdown("##### ⚖️ Final Decision")
                                    decision = st.selectbox("Decision", ["Approve", "Compromise", "Reject"], key=f"d_{row['Lot_ID']}")
                                    note = st.text_input("Note to Customer", key=f"n_{row['Lot_ID']}")
                                    if st.button("🏁 Close Case", key=f"btn_{row['Lot_ID']}", type="primary"):
                                        update_status(row['Lot_ID'], "Case Closed", f"MCS: {decision}", "Completed", decision, note)
                                        st.rerun()
                                st.markdown("---")
                                st.caption("🛠️ **Override**")
                                new_handler = st.selectbox("Re-assign to:", ["QC", "R&D", "Logistics", "Marketing & Customer Service (MCS)"], key=f"move_{row['Lot_ID']}")
                                if st.button("Force Move", key=f"btn_move_{row['Lot_ID']}"):
                                    update_status(row['Lot_ID'], f"Re-assigned to {new_handler}", "MCS moved case", force_handler=new_handler)
                                    st.success(f"Moved to {new_handler}")
                                    st.rerun()
                            else: 
                                note = st.text_input("Investigation Note", key=f"in_{row['Lot_ID']}")
                                if st.button("➡️ Forward to MCS", key=f"fwd_{row['Lot_ID']}"):
                                    update_status(row['Lot_ID'], "Investigation Complete", f"{user_dept}: {note}", "Marketing & Customer Service (MCS)")
                                    st.rerun()
            else:
                st.success(f"🎉 No pending tasks for **{user_dept}**")
        with subtab_history:
            st.dataframe(completed_tasks, use_container_width=True)

# --- TAB 4: Tracking ---
with tab4:
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
