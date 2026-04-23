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

    def _assign_dynamic_blocks(self, df):
        if df.empty: return df
        df = df.sort_values('Dateofweld')
        groups = df.groupby(self.lot_criteria)
        processed_chunks = []

        for _, group in groups:
            group = group.copy()
            block_ids = []
            if not group.empty:
                start_date = group['Dateofweld'].iloc[0]
                current_block = 0
                for date in group['Dateofweld']:
                    if pd.isna(date):
                        block_ids.append(-1)
                    elif date < start_date + pd.Timedelta(days=self.window_days):
                        block_ids.append(current_block)
                    else:
                        start_date = date
                        current_block += 1
                        block_ids.append(current_block)
            group['Block_ID'] = block_ids
            processed_chunks.append(group)
        
        return pd.concat(processed_chunks) if processed_chunks else df

    def get_lot_audit(self, df):
        diag_info = {}
        d = df.copy()
        diag_info['raw_count'] = len(d)
        
        d['Dateofweld'] = pd.to_datetime(d['Dateofweld'], errors='coerce')
        d['RTDate1'] = pd.to_datetime(d['RTDate1'], errors='coerce')
        
        d_no_date = d[d['Dateofweld'].isna()]
        d = d.dropna(subset=['Dateofweld'])
        diag_info['dropped_no_date'] = len(d_no_date)
        
        if d.empty: return pd.DataFrame(), pd.DataFrame(), diag_info
        
        d['RT_Perc'] = pd.to_numeric(d['RT_Perc'], errors='coerce').replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        
        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        allowed_values = location_map.get(self.scope, [])
        
        d_wrong_scope = d[~d['location'].isin(allowed_values)]
        diag_info['dropped_wrong_scope'] = len(d_wrong_scope)
        
        d_mandatory = d[(d['location'].isin(allowed_values)) & (d['RT_Perc'] >= 100)]
        diag_info['dropped_mandatory'] = len(d_mandatory)
        
        df_filtered = d[(d['location'].isin(allowed_values)) & (d['RT_Perc'] < 100)].copy()
        diag_info['final_pool'] = len(df_filtered)
        
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame(), diag_info

        df_with_blocks = self._assign_dynamic_blocks(df_filtered)
        
        def build_lot_id(row):
            parts = [str(row['Block_ID'])]
            for criterion in self.lot_criteria:
                parts.append(str(row[criterion]) if criterion in row else "NA")
            return "_".join(parts)

        df_with_blocks['Lot_ID'] = df_with_blocks.apply(build_lot_id, axis=1)

        audit = df_with_blocks.groupby('Lot_ID').agg(
            Total_Joints=('Joint_ID', 'count'), 
            Current_RT_Done=('RTDate1', 'count'),
            Current_RT_Req=('RT_Perc', 'max'),    
            Welder=('Welder1', 'first'),
            Process=('WPS.1.Description', 'first'),
            Material=('MaterialType', 'first'),
            Subcontractor=('Subc', 'first'),
            Block_Start_Date=('Dateofweld', 'min')
        ).reset_index()
        
        audit['Current_RT_Done_%'] = (audit['Current_RT_Done'] / audit['Total_Joints'] * 100).round(1)
        audit['Required'] = np.ceil(audit['Total_Joints'] * (audit['Current_RT_Req'] / 100)).astype(int)
        audit['Deficit'] = (audit['Required'] - audit['Current_RT_Done']).clip(lower=0)
        audit['Status'] = np.where(audit['Deficit'] > 0, '🔴 OPEN', '🟢 CLOSED')
        
        cols_order = ['Status', 'Lot_ID', 'Subcontractor', 'Total_Joints', 'Current_RT_Done', 'Current_RT_Done_%', 
                      'Current_RT_Req', 'Required', 'Deficit', 'Welder', 'Process', 'Material', 'Block_Start_Date']
        return audit[cols_order], df_with_blocks, diag_info

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
def get_available_scopes(df, selected_sub):
    location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
    # Si es ALL, miramos todas las locaciones del archivo. Si no, solo las del subcontratista.
    if selected_sub == "ALL":
        locs = df['location'].unique()
    else:
        locs = df[df['Subc'] == selected_sub]['location'].unique()
    
    available = []
    for scope, vals in location_map.items():
        if any(loc in locs for loc in vals): available.append(scope)
    return available

# --- USER INTERFACE ---
st.set_page_config(page_title="RT Optimizer", layout="wide")
st.title("🛡️ RT Optimizer")

st.sidebar.header(":material/settings: Configuration")

uploaded_file = st.file_uploader("Upload Daily SQL Extraction (CSV)", type="csv")

if uploaded_file:
    df_input = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
    df_input.columns = df_input.columns.str.strip()
    
    for col in df_input.select_dtypes(['object']).columns:
        df_input[col] = df_input[col].astype(str).str.strip()

    required_cols = ['Joint_ID', 'Subc', 'Welder1', 'Line', 'location', 'MaterialType', 'WPS.1.Description', 'Dateofweld', 'RTDate1', 'RT_Perc']
    new_cols = {col: req for col in df_input.columns for req in required_cols if col.lower() == req.lower()}
    df_input.rename(columns=new_cols, inplace=True)
    
    if not all(c in df_input.columns for c in required_cols):
        st.error(f"❌ Missing columns. Required: {required_cols}")
    else:
        # --- SIDEBAR DINÁMICO ---
        subs_list = ["ALL"] + sorted(df_input['Subc'].unique().tolist())
        selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
        
        available_scopes = get_available_scopes(df_input, selected_sub)
        
        if available_scopes:
            location_scope = st.sidebar.radio(f"Location Scope:", options=available_scopes)
            days_per_lot = st.sidebar.number_input("Days per Window", min_value=1, value=14)
            fallback_percentage = st.sidebar.slider("Fallback RT %", 0, 100, 10)
            
            st.sidebar.divider()
            c_subc = st.sidebar.checkbox("Subcontractor (Subc)", value=True)
            c_welder = st.sidebar.checkbox("Welder ID (Welder1)", value=True)
            c_material = st.sidebar.checkbox("Material Type (MaterialType)", value=True)
            c_process = st.sidebar.checkbox("Welding Process (WPS.1.Description)", value=True)
            c_line = st.sidebar.checkbox("Line ID (Line)", value=False)
            db_criteria_map = [col for cond, col in zip([c_subc, c_welder, c_material, c_process, c_line], ['Subc', 'Welder1', 'MaterialType', 'WPS.1.Description', 'Line']) if cond]

            # --- FILTRADO DE DATOS ANTES DEL MOTOR ---
            df_to_process = df_input.copy()
            if selected_sub != "ALL":
                df_to_process = df_to_process[df_to_process['Subc'] == selected_sub]

            engine = RTOptimizerEngine(fallback_percentage/100, days_per_lot, db_criteria_map, location_scope)
            audit_df, df_with_lots, diagnostic = engine.get_lot_audit(df_to_process)

            if audit_df.empty:
                st.warning(f"No sampling lots found for {selected_sub} in {location_scope}.")
            else:
                tab1, tab2, tab3 = st.tabs(["📋 Work Order", "📊 Dashboard", "🛠️ Diagnostics"])

                with tab1:
                    st.subheader(f"Inspection Plan: {selected_sub}")
                    if st.button("🚀 Generate Plan"):
                        result = engine.execute_optimization(df_with_lots, audit_df.copy())
                        if not result.empty:
                            result['Plan_Date'] = datetime.now().strftime('%d/%m/%Y')
                            st.write(f"Recommended Inspections: **{len(result)}**")
                            st.dataframe(result, use_container_width=True, hide_index=True)
                            st.download_button("📥 Download", result.to_csv(sep=';', index=False).encode('utf-8-sig'), f"plan_{selected_sub}.csv", "text/csv")
                        else:
                            st.success("✅ Compliance achieved.")

                with tab2:
                    # KPIs basados directamente en el audit_df calculado (ya filtrado por sidebar)
                    k1, k2, k3, k4, k5 = st.columns(5)
                    total_l = len(audit_df); open_l = len(audit_df[audit_df['Status'] == '🔴 OPEN'])
                    k1.metric("Total Lots", total_l); k2.metric("Open Lots", open_l)
                    k3.metric("Project Compliance", f"{((total_l - open_l)/total_l)*100:.1f}%" if total_l > 0 else "0%")
                    k4.metric("Avg. Actual RT %", f"{audit_df['Current_RT_Done_%'].mean():.1f}%" if total_l > 0 else "0%")
                    k5.metric("Avg. Target RT %", f"{audit_df['Current_RT_Req'].mean():.1f}%" if total_l > 0 else "0%")
                    
                    st.divider()
                    # Filtros de tabla (ahora actúan sobre el set seleccionado en sidebar)
                    f1, f2, f3 = st.columns(3)
                    with f1: s_w = st.multiselect("Filter Welder", options=sorted(audit_df['Welder'].unique()))
                    with f2: s_m = st.multiselect("Filter Material", options=sorted(audit_df['Material'].unique()))
                    with f3: s_s = st.multiselect("Filter Status", options=['🔴 OPEN', '🟢 CLOSED'])

                    f_audit = audit_df.copy()
                    if s_w: f_audit = f_audit[f_audit['Welder'].isin(s_w)]
                    if s_m: f_audit = f_audit[f_audit['Material'].isin(s_m)]
                    if s_s: f_audit = f_audit[f_audit['Status'].isin(s_s)]

                    event = st.dataframe(f_audit, use_container_width=True, on_select="rerun", selection_mode="single-row", hide_index=True)
                    if event.selection.rows:
                        row_idx = event.selection.rows[0]; lot_id = f_audit.iloc[row_idx]['Lot_ID']
                        st.markdown(f"### 🔍 Detailed Explorer: Lot `{lot_id}`")
                        st.dataframe(df_with_lots[df_with_lots['Lot_ID'] == lot_id], use_container_width=True, hide_index=True)

                with tab3:
                    st.subheader("Data Processing Diagnostics")
                    st.write(f"Metrics for **{selected_sub}** in **{location_scope}** scope.")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Inflow:**")
                        st.write(f"- Total joints after subcontractor filter: {diagnostic['raw_count']}")
                        st.write(f"- Dropped (No Dateofweld): {diagnostic['dropped_no_date']}")
                        st.write(f"- Dropped (Other Subcontractor Scope): {diagnostic['dropped_wrong_scope']}")
                        st.write(f"- Dropped (RT_Perc = 100%): {diagnostic['dropped_mandatory']}")
                    with c2:
                        st.write("**Result:**")
                        st.success(f"- Final pool for sampling: {diagnostic['final_pool']}")
        else:
            st.sidebar.warning("No valid locations found in CSV.")

else:
    st.info("💡 Please upload your SQL CSV extraction.")
    # (Schema table code kept internal)