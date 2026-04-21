import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- OPTIMIZATION ENGINE ---
class RTOptimizerEngine:
    def __init__(self, fallback_rt_perc, window_days, lot_criteria, selected_location_scope):
        self.fallback_rt_perc = fallback_rt_perc
        self.window_days = window_days
        self.lot_criteria = lot_criteria 
        self.scope = selected_location_scope

    def get_lot_audit(self, df):
        # 1. Sanitization
        df['Dateofweld'] = pd.to_datetime(df['Dateofweld'], dayfirst=True, errors='coerce')
        df['RTDate1'] = pd.to_datetime(df['RTDate1'], dayfirst=True, errors='coerce')
        df['RT_Perc'] = pd.to_numeric(df['RT_Perc'], errors='coerce')
        
        # --- LÓGICA DE FALLBACK CORREGIDA ---
        # Tratamos el 0 y el Nulo como "información faltante" y aplicamos el Fallback
        df['RT_Perc'] = df['RT_Perc'].replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        
        # 2. Location Mapping
        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        allowed_values = location_map.get(self.scope, [])
        
        # Filtramos por Scope y solo quitamos las de 100% (Las de 0% ahora tienen el Fallback)
        df_filtered = df[
            (df['location'].isin(allowed_values)) & 
            (df['RT_Perc'] < 100)
        ].copy()
        
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame()

        # 3. Time Blocks
        start_date = df_filtered['Dateofweld'].min()
        df_filtered['Block_ID'] = ((df_filtered['Dateofweld'] - start_date).dt.days // self.window_days).fillna(-1).astype(int)
        
        # 4. Lot ID Building
        def build_lot_id(row):
            parts = [str(row['Block_ID'])]
            for criterion in self.lot_criteria:
                parts.append(str(row[criterion]))
            return "_".join(parts)

        df_audit_base = df_filtered[df_filtered['Block_ID'] >= 0].copy()
        df_audit_base['Lot_ID'] = df_audit_base.apply(build_lot_id, axis=1)

        # 5. Grouping
        audit = df_audit_base.groupby('Lot_ID').agg(
            Total_Joints=('Joint_ID', 'count'), 
            Current_RT_Done=('RTDate1', 'count'),
            Current_RT_Req=('RT_Perc', 'max'),    
            Welder=('Welder1', 'first'),
            Process=('WPS.1.Description', 'first'),
            Material=('MaterialType', 'first'),
            Subcontractor=('Subc', 'first'),
            Block=('Block_ID', 'first')
        ).reset_index()
        
        audit['Current_RT_Done_%'] = (audit['Current_RT_Done'] / audit['Total_Joints'] * 100).round(1)
        audit['Required'] = np.ceil(audit['Total_Joints'] * (audit['Current_RT_Req'] / 100)).astype(int)
        audit['Deficit'] = (audit['Required'] - audit['Current_RT_Done']).clip(lower=0)
        audit['Status'] = np.where(audit['Deficit'] > 0, '🔴 OPEN', '🟢 CLOSED')
        
        cols_order = ['Status', 'Lot_ID', 'Subcontractor', 'Total_Joints', 'Current_RT_Done', 'Current_RT_Done_%', 
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

# --- UTILS ---
def get_available_scopes_for_sub(df, sub_name):
    ws_vals = ["YWS", "S", "WS"]
    fw_vals = ["YFW", "FW", "F"]
    pl_vals = ["PL"]
    sub_locs = df[df['Subc'] == sub_name]['location'].unique()
    available = []
    if any(loc in sub_locs for loc in ws_vals): available.append("WS")
    if any(loc in sub_locs for loc in fw_vals): available.append("FW")
    if any(loc in sub_locs for loc in pl_vals): available.append("PL")
    return available

# --- USER INTERFACE ---
st.set_page_config(page_title="RT Optimizer", layout="wide")
st.title(":material/engineering: RT Optimizer")

# 1. SIDEBAR
st.sidebar.header(":material/settings: Configuration")

uploaded_file = st.file_uploader("Upload Daily SQL Extraction (CSV)", type="csv")

if uploaded_file:
    df_input = pd.read_csv(uploaded_file, sep=';')
    df_input.columns = df_input.columns.str.strip()
    for col in df_input.select_dtypes(['object']).columns:
        df_input[col] = df_input[col].astype(str).str.strip()

    if 'Subc' in df_input.columns and 'location' in df_input.columns:
        subs_list = sorted(df_input['Subc'].unique())
        selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
        
        available_scopes = get_available_scopes_for_sub(df_input, selected_sub)
        if available_scopes:
            location_scope = st.sidebar.radio(
                f"Location Scope for {selected_sub}:", 
                options=available_scopes,
                help="Only locations where this subcontractor has registered joints are shown."
            )
        else:
            st.sidebar.warning("No valid locations found for this Subcontractor.")
            st.stop()
    else:
        st.error("CSV must contain 'Subc' and 'location' columns.")
        st.stop()

    days_per_lot = st.sidebar.number_input(
        "Days per Window", 
        min_value=1, value=14,
        help="Period of time which defines a Block"
    )
    
    fallback_percentage = st.sidebar.slider(
        "Fallback RT %", 
        0, 100, 10,
        help="This value is applied if 'RT_Perc' is empty OR 0 in the CSV file."
    )
    st.sidebar.caption("⚠️ *Fallback acts as a default for uninitialized data (0 or Null).*")

    st.sidebar.divider()
    st.sidebar.subheader("Lot Identity Factors")
    c_subc = st.sidebar.checkbox("Subcontractor", value=True)
    c_welder = st.sidebar.checkbox("Welder ID", value=True)
    c_material = st.sidebar.checkbox("Material Type", value=True)
    c_process = st.sidebar.checkbox("Welding Process", value=True)
    c_line = st.sidebar.checkbox("Line", value=False)

    db_criteria_map = []
    if c_subc: db_criteria_map.append('Subc')
    if c_welder: db_criteria_map.append('Welder1')
    if c_material: db_criteria_map.append('MaterialType')
    if c_process: db_criteria_map.append('WPS.1.Description')
    if c_line: db_criteria_map.append('Line')

    engine = RTOptimizerEngine(fallback_percentage/100, days_per_lot, db_criteria_map, location_scope)
    audit_df, df_with_lots = engine.get_lot_audit(df_input)

    if audit_df.empty:
        st.warning(f"⚠️ No joints found for scope '{location_scope}' (All are 100% or excluded).")
    else:
        tab1, tab2 = st.tabs(["📋 Work Order Generator", "📊 Dashboard & Lot Explorer"])

        with tab1:
            st.subheader(f"Inspection Plan: {selected_sub} | Scope: {location_scope}")
            sub_df_lots = df_with_lots[df_with_lots['Subc'] == selected_sub]
            sub_audit = audit_df[audit_df['Subcontractor'] == selected_sub]
            
            if st.button(f"🚀 Generate Inspection Plan"):
                with st.spinner('Optimizing synergies...'):
                    result = engine.execute_optimization(sub_df_lots, sub_audit.copy())
                    if not result.empty:
                        st.write(f"Recommended Inspections: **{len(result)}**")
                        display_cols = ['Joint_ID', 'Subc', 'Welder1', 'Line', 'location', 'MaterialType', 'WPS.1.Description', 'Dateofweld', 'RT_Perc']
                        st.dataframe(result[[c for c in display_cols if c in result.columns]], use_container_width=True, hide_index=True)
                        csv_data = result.to_csv(sep=';', index=False).encode('utf-8-sig')
                        st.download_button("📥 Download Plan (CSV)", csv_data, f"plan_{selected_sub}_{location_scope}.csv", "text/csv")
                    else:
                        st.success("✅ Compliance achieved for this scope.")

        with tab2:
            k1, k2, k3, k4 = st.columns(4)
            total_l = len(dash_audit := (audit_df.copy() if (view_option := st.selectbox("📊 Dashboard View Scope:", options=["ALL"] + subs_list, index=subs_list.index(selected_sub)+1)) == "ALL" else audit_df[audit_df['Subcontractor'] == view_option]))
            open_l = len(dash_audit[dash_audit['Status'] == '🔴 OPEN'])
            k1.metric("Total Lots", total_l)
            k2.metric("Open Lots", open_l)
            k3.metric("Closed Lots", total_l - open_l)
            k4.metric("Compliance", f"{((total_l - open_l)/total_l)*100:.1f}%" if total_l > 0 else "0%")

            st.divider()
            f1, f2, f3, f4 = st.columns(4)
            with f1: s_welders = st.multiselect("Filter Welder", options=sorted(dash_audit['Welder'].unique()))
            with f2: s_materials = st.multiselect("Filter Material", options=sorted(dash_audit['Material'].unique()))
            with f3: s_processes = st.multiselect("Filter Process", options=sorted(dash_audit['Process'].unique()))
            with f4: s_status = st.multiselect("Filter Status", options=['🔴 OPEN', '🟢 CLOSED'])

            filtered_audit = dash_audit.copy()
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
                joints_in_lot = df_with_lots[df_with_lots['Lot_ID'] == lot_id]
                st.dataframe(joints_in_lot[['Joint_ID', 'Subc', 'Welder1', 'Line', 'Dateofweld', 'RTDate1', 'RT_Perc']], use_container_width=True, hide_index=True)
            else:
                st.caption("💡 Click on a row above to see individual joints.")

else:
    st.info("💡 **Awaiting Data.** Please upload your SQL CSV extraction to begin.")
    st.markdown("### Required CSV Data Structure")
    example_schema = {
        'Column Name': ['Joint_ID', 'Subc', 'Welder1', 'Line', 'location', 'MaterialType', 'WPS.1.Description', 'Dateofweld', 'RTDate1', 'RT_Perc'],
        'Description': ['Unique ID', 'Subcontractor', 'Welder ID', 'Line Number', 'Location Type', 'Material Group', 'Welding Process', 'DD/MM/YYYY', 'Empty if pending', 'Target % (5, 10, 100)']
    }
    st.table(pd.DataFrame(example_schema))
    st.markdown("""
    **Scope Rules:**
    *   `RT_Perc` = 100: Mandatory inspections (automatically excluded from lot sampling).
    *   `RT_Perc` = 0 or Null: Missing info. **Fallback RT %** value from sidebar will be applied.
    """)