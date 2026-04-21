import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- OPTIMIZATION ENGINE ---
class RTOptimizerEngine:
    def __init__(self, fallback_rt_perc, window_days, lot_criteria):
        self.fallback_rt_perc = fallback_rt_perc
        self.window_days = window_days
        self.lot_criteria = lot_criteria 

    def get_lot_audit(self, df):
        df['Dateofweld'] = pd.to_datetime(df['Dateofweld'], dayfirst=True, errors='coerce')
        df['RTDate1'] = pd.to_datetime(df['RTDate1'], dayfirst=True, errors='coerce')
        df['RT_Perc'] = pd.to_numeric(df['RT_Perc'], errors='coerce')
        
        # Aplicación del Fallback: si la celda está vacía, usa el valor del slider
        df['RT_Perc'] = df['RT_Perc'].fillna(self.fallback_rt_perc * 100)
        
        df_filtered = df[(df['RT_Perc'] > 0) & (df['RT_Perc'] < 100)].copy()
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame()

        start_date = df_filtered['Dateofweld'].min()
        df_filtered['Block_ID'] = ((df_filtered['Dateofweld'] - start_date).dt.days // self.window_days).fillna(-1).astype(int)
        
        def build_lot_id(row):
            parts = [str(row['Block_ID'])]
            for criterion in self.lot_criteria:
                parts.append(str(row[criterion]))
            return "_".join(parts)

        df_audit_base = df_filtered[df_filtered['Block_ID'] >= 0].copy()
        df_audit_base['Lot_ID'] = df_audit_base.apply(build_lot_id, axis=1)

        audit = df_audit_base.groupby('Lot_ID').agg(
            Total_Joints=('Joint_ID', 'count'), 
            Current_RT_Done=('RTDate1', 'count'),
            Current_RT_Req=('RT_Perc', 'max'),    
            Welder=('Welder1', 'first'),
            Process=('WPS.1.Description', 'first'),
            Material=('MaterialType', 'first'),
            Block=('Block_ID', 'first')
        ).reset_index()
        
        audit['Current_RT_Done_%'] = (audit['Current_RT_Done'] / audit['Total_Joints'] * 100).round(1)
        audit['Required'] = np.ceil(audit['Total_Joints'] * (audit['Current_RT_Req'] / 100)).astype(int)
        audit['Deficit'] = (audit['Required'] - audit['Current_RT_Done']).clip(lower=0)
        audit['Status'] = np.where(audit['Deficit'] > 0, '🔴 OPEN', '🟢 CLOSED')
        
        cols_order = ['Status', 'Lot_ID', 'Total_Joints', 'Current_RT_Done', 'Current_RT_Done_%', 
                      'Current_RT_Req', 'Required', 'Deficit', 'Welder', 'Process', 'Material', 'Block']
        return audit[cols_order], df_audit_base

    def execute_optimization(self, df_audit_base, audit):
        if audit.empty: return pd.DataFrame()
        debts = audit.set_index('Lot_ID')['Deficit'].to_dict()
        candidates = df_audit_base[df_audit_base['RTDate1'].isnull()].copy()
        inspection_plan = []
        
        while sum(debts.values()) > 0 and not candidates.empty:
            def calc_impact(row):
                l_id = row['Lot_ID']
                impact = 1 if debts.get(l_id, 0) > 0 else 0
                if 'WPS.1.Description' in self.lot_criteria and 'GTAW+SMAW' in l_id:
                    target_gtaw = l_id.replace('GTAW+SMAW', 'GTAW')
                    if debts.get(target_gtaw, 0) > 0: impact += 0.5 
                return impact

            candidates['Impact'] = candidates.apply(calc_impact, axis=1)
            if candidates['Impact'].max() == 0: break
            best_idx = candidates['Impact'].idxmax()
            selected_joint = candidates.loc[best_idx]
            l_id = selected_joint['Lot_ID']
            if debts.get(l_id, 0) > 0: debts[l_id] -= 1
            if 'WPS.1.Description' in self.lot_criteria and 'GTAW+SMAW' in l_id:
                target_gtaw = l_id.replace('GTAW+SMAW', 'GTAW')
                if debts.get(target_gtaw, 0) > 0: debts[target_gtaw] -= 1
            inspection_plan.append(selected_joint)
            candidates = candidates.drop(best_idx)
        return pd.DataFrame(inspection_plan)

# --- USER INTERFACE ---
st.set_page_config(page_title="RT Optimizer", layout="wide")
st.title("🛡️ RT Optimizer") # Nombre actualizado

# 1. SIDEBAR CON EXPLICACIONES
st.sidebar.header("⚙️ Configuration")

# Explicación de Window
days_per_lot = st.sidebar.number_input(
    "Days per Window", 
    min_value=1, 
    value=14,
    help="Maximum time period allowed for a lot to remain open according to ASME B31.3."
)

# Explicación de Fallback (Tu petición)
fallback_percentage = st.sidebar.slider(
    "Fallback RT Requirement %", 
    0, 100, 10,
    help="This percentage is used ONLY if a joint in the CSV has an empty 'RT_Perc' cell. It acts as a safety default value."
)
st.sidebar.caption("⚠️ *The fallback is only applied to missing data in the source file.*")

st.sidebar.divider()

st.sidebar.subheader("Lot Identity Factors")
st.sidebar.info("Select the variables that define a unique Designated Lot:")
c_welder = st.sidebar.checkbox("Welder ID", value=True)
c_material = st.sidebar.checkbox("Material Type", value=True)
c_process = st.sidebar.checkbox("Welding Process", value=True)
c_line = st.sidebar.checkbox("Line ID", value=False)

db_criteria_map = []
if c_welder: db_criteria_map.append('Welder1')
if c_material: db_criteria_map.append('MaterialType')
if c_process: db_criteria_map.append('WPS.1.Description')
if c_line: db_criteria_map.append('Line')

# 2. FILE UPLOADER
uploaded_file = st.file_uploader("Upload Daily SQL Extraction (CSV)", type="csv")

if uploaded_file:
    df_input = pd.read_csv(uploaded_file, sep=';')
    df_input.columns = df_input.columns.str.strip()
    for col in df_input.select_dtypes(['object']).columns:
        df_input[col] = df_input[col].astype(str).str.strip()

    engine = RTOptimizerEngine(fallback_percentage/100, days_per_lot, db_criteria_map)
    audit_df, df_with_lots = engine.get_lot_audit(df_input)

    if audit_df.empty:
        st.warning("⚠️ No random sampling lots found (all joints are 0% or 100%).")
    else:
        tab1, tab2 = st.tabs(["📋 Work Order Generator", "📊 Dashboard & Lot Explorer"])

        with tab1:
            st.subheader("Plan Optimization")
            if st.button("🚀 Calculate Optimized Plan"):
                result = engine.execute_optimization(df_with_lots, audit_df.copy())
                if not result.empty:
                    st.write(f"Recommended Inspections: **{len(result)}**")
                    display_cols = ['Joint_ID', 'Welder1', 'Line', 'MaterialType', 'WPS.1.Description', 'Dateofweld', 'RT_Perc']
                    actual_display = [c for c in display_cols if c in result.columns]
                    st.dataframe(result[actual_display], use_container_width=True)
                    csv_data = result.to_csv(sep=';', index=False).encode('utf-8-sig')
                    st.download_button("📥 Download Plan (CSV)", csv_data, "work_order.csv", "text/csv")
                else:
                    st.success("✅ Everything is in compliance.")

        with tab2:
            k1, k2, k3, k4 = st.columns(4)
            total_l = len(audit_df)
            open_l = len(audit_df[audit_df['Status'] == '🔴 OPEN'])
            k1.metric("Total Lots", total_l)
            k2.metric("Open Lots", open_l)
            k3.metric("Closed Lots", total_l - open_l)
            k4.metric("Compliance", f"{((total_l - open_l)/total_l)*100:.1f}%")

            st.divider()
            st.subheader("🔍 Filter Summary Table")
            f1, f2, f3, f4 = st.columns(4)
            with f1: s_welders = st.multiselect("Welder", options=sorted(audit_df['Welder'].unique()))
            with f2: s_materials = st.multiselect("Material", options=sorted(audit_df['Material'].unique()))
            with f3: s_processes = st.multiselect("Process", options=sorted(audit_df['Process'].unique()))
            with f4: s_status = st.multiselect("Status", options=['🔴 OPEN', '🟢 CLOSED'])

            filtered_audit = audit_df.copy()
            if s_welders: filtered_audit = filtered_audit[filtered_audit['Welder'].isin(s_welders)]
            if s_materials: filtered_audit = filtered_audit[filtered_audit['Material'].isin(s_materials)]
            if s_processes: filtered_audit = filtered_audit[filtered_audit['Process'].isin(s_processes)]
            if s_status: filtered_audit = filtered_audit[filtered_audit['Status'].isin(s_status)]

            selection_event = st.dataframe(filtered_audit, use_container_width=True, on_select="rerun", selection_mode="single-row", hide_index=True)

            if selection_event.selection.rows:
                row_idx = selection_event.selection.rows[0]
                selected_lot_data = filtered_audit.iloc[row_idx]
                lot_id = selected_lot_data['Lot_ID']
                st.markdown(f"### 🔍 Detailed Explorer: Lot `{lot_id}`")
                det_c1, det_c2, det_c3, det_c4 = st.columns(4)
                det_c1.write(f"**Welder:** {selected_lot_data['Welder']}")
                det_c2.write(f"**Material:** {selected_lot_data['Material']}")
                det_c3.write(f"**Process:** {selected_lot_data['Process']}")
                det_c4.write(f"**Status:** {selected_lot_data['Status']}")
                st.info(f"Progress: {selected_lot_data['Current_RT_Done']} of {selected_lot_data['Required']} RTs done ({selected_lot_data['Current_RT_Done_%']}%)")
                joints_in_lot = df_with_lots[df_with_lots['Lot_ID'] == lot_id]
                st.dataframe(joints_in_lot[['Joint_ID', 'Line', 'Dateofweld', 'RTDate1', 'RT_Perc']], use_container_width=True, hide_index=True)
            else:
                st.caption("💡 Click on any row above to see individual joints.")

else:
    st.info("💡 Please upload your SQL CSV extraction to begin.")