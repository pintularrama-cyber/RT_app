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
                    if pd.isna(date): block_ids.append(-1)
                    elif date < start_date + pd.Timedelta(days=self.window_days): block_ids.append(current_block)
                    else:
                        start_date = date
                        current_block += 1
                        block_ids.append(current_block)
            group['Block_ID'] = block_ids
            processed_chunks.append(group)
        return pd.concat(processed_chunks) if processed_chunks else df

    def get_lot_audit(self, df):
        d = df.copy()
        d['Dateofweld'] = pd.to_datetime(d['Dateofweld'], errors='coerce')
        d['RTDate1'] = pd.to_datetime(d['RTDate1'], errors='coerce')
        d['RT2Date1'] = pd.to_datetime(d.get('RT2Date1', None), errors='coerce')
        d = d.dropna(subset=['Dateofweld'])
        if d.empty: return pd.DataFrame(), pd.DataFrame()
        
        d['RT_Perc'] = pd.to_numeric(d['RT_Perc'], errors='coerce').replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        
        for col in ['RT1rej', 'RT2rej']:
            if col in d.columns:
                d[col] = d[col].astype(str).str.upper().str.strip()
                d[col] = d[col].map({'TRUE': True, 'FALSE': False, '1': True, '0': False, 'NAN': False}).fillna(False)
            else: d[col] = False

        if self.scope == "ALL":
            df_filtered = d[d['RT_Perc'] < 100].copy()
        else:
            location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
            allowed_values = location_map.get(self.scope, [])
            df_filtered = d[(d['location'].isin(allowed_values)) & (d['RT_Perc'] < 100)].copy()
        
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame()
        df_with_blocks = self._assign_dynamic_blocks(df_filtered)
        
        def build_lot_id(row):
            parts = [str(row['Block_ID'])]
            for criterion in self.lot_criteria:
                parts.append(str(row[criterion]) if criterion in row else "NA")
            return "_".join(parts)

        df_with_blocks['Lot_ID'] = df_with_blocks.apply(build_lot_id, axis=1)

        audit = df_with_blocks.groupby('Lot_ID').agg(
            Total_Joints=('Joint_ID', 'count'), 
            RT1_Done_Count=('RTDate1', 'count'),
            RT1_Rejects=('RT1rej', 'sum'),
            RT2_Pending_Count=('RT1rej', lambda x: ((x == True) & (df_with_blocks.loc[x.index, 'RT2Date1'].isna())).sum()),
            Current_RT_Req=('RT_Perc', 'max'),    
            Welder=('Welder1', 'first'),
            Process=('WPS.1.Description', 'first'),
            Material=('MaterialType', 'first'),
            Subcontractor=('Subc', 'first'),
            location=('location', 'first'),
            Block_Start_Date=('Dateofweld', 'min')
        ).reset_index()
        
        audit['Current_RT_Done'] = audit['RT1_Done_Count'] - audit['RT1_Rejects']
        
        def calculate_required(row):
            base_req = np.ceil(row['Total_Joints'] * (row['Current_RT_Req'] / 100))
            if row['RT1_Rejects'] == 1: return min(row['Total_Joints'], base_req + 2)
            elif row['RT1_Rejects'] > 1: return row['Total_Joints']
            return base_req

        audit['Required'] = audit.apply(calculate_required, axis=1).astype(int)
        audit['Current_RT_Done_%'] = (audit['Current_RT_Done'] / audit['Total_Joints'] * 100).round(1)
        audit['Deficit'] = (audit['Required'] - audit['Current_RT_Done']).clip(lower=0)
        audit['Status'] = np.where((audit['Deficit'] == 0) & (audit['RT2_Pending_Count'] == 0), '🟢 CLOSED', '🔴 OPEN')
        
        # --- LÓGICA DE ETIQUETADO CORREGIDA ---
        rejects_map = audit.set_index('Lot_ID')['RT1_Rejects'].to_dict()
        def label_joint_type(row):
            l_id = row['Lot_ID']
            n_rej = rejects_map.get(l_id, 0)
            
            # 1. Prioridad: ¿Se ha reparado ya?
            if row['RT1rej'] and pd.notna(row['RT2Date1']): return "🛠️ REPAIR DONE (RT2)"
            # 2. ¿Está rechazada y pendiente?
            if row['RT1rej']: return "❌ REJECTED"
            # 3. ¿Es una estándar completada?
            if pd.notna(row['RTDate1']): return "✅ STANDARD (OK)"
            # 4. Pendientes (Penalty o Full Audit)
            if n_rej == 1: return "🚨 PENALTY (Tracer)"
            if n_rej > 1: return "🧨 FULL AUDIT (100%)"
            return "Standard Sampling"

        df_with_blocks['Inspection_Type'] = df_with_blocks.apply(label_joint_type, axis=1)

        cols_order = ['Status', 'Lot_ID', 'Subcontractor', 'Total_Joints', 'Current_RT_Done', 'Current_RT_Done_%', 
                      'RT1_Rejects', 'RT2_Pending_Count', 'Current_RT_Req', 'Required', 'Deficit', 'Welder', 'Process', 'Material', 'location', 'Block_Start_Date']
        return audit[cols_order], df_with_blocks

    def execute_optimization(self, df_audit_base, audit):
        if audit.empty: return pd.DataFrame()
        debts = audit.set_index('Lot_ID')['Deficit'].to_dict()
        penalty_status = audit.set_index('Lot_ID')['RT1_Rejects'].to_dict()
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
            selected_joint = candidates.loc[best_idx].copy()
            l_id = selected_joint['Lot_ID']
            n_rej = penalty_status.get(l_id, 0)
            selected_joint['Inspection_Reason'] = "🚨 PENALTY" if n_rej == 1 else ("🧨 FULL AUDIT" if n_rej > 1 else "Standard")
            if debts.get(l_id, 0) > 0: debts[l_id] -= 1
            inspection_plan.append(selected_joint)
            candidates = candidates.drop(best_idx)
        return pd.DataFrame(inspection_plan)

# --- UTILS ---
def get_available_scopes(df, subcontractor):
    location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
    if subcontractor == "ALL": locs = df['location'].unique()
    else: locs = df[df['Subc'] == subcontractor]['location'].unique()
    available = []
    for scope, vals in location_map.items():
        if any(loc in locs for loc in vals): available.append(scope)
    return ["ALL"] + available if len(available) > 1 else available

# --- UI ---
st.set_page_config(page_title="RT Optimizer", layout="wide", page_icon="🏗️")
st.title(":material/engineering: RT Optimizer")

st.sidebar.header("⚙️ Configuration")
uploaded_file = st.file_uploader("Upload Daily SQL Extraction (CSV)", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
    df_raw.columns = df_raw.columns.str.strip()
    required_cols = ['Joint_ID', 'Subc', 'Welder1', 'Line', 'location', 'MaterialType', 'WPS.1.Description', 'Dateofweld', 'RTDate1', 'RT_Perc', 'RT1rej', 'RT2Date1', 'RT2rej']
    new_cols = {col: req for col in df_raw.columns for req in required_cols if col.lower() == req.lower()}
    df_raw.rename(columns=new_cols, inplace=True)
    for col in df_raw.select_dtypes(['object']).columns: df_raw[col] = df_raw[col].astype(str).str.strip()

    subs_list = ["ALL"] + sorted(df_raw['Subc'].unique().tolist())
    selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
    scopes = get_available_scopes(df_raw, selected_sub)
    location_scope = st.sidebar.radio("Location Scope:", options=scopes, index=0)
    days_per_lot = st.sidebar.number_input("Days per Window", min_value=1, value=14)
    fallback_perc = st.sidebar.slider("Fallback RT %", 0, 100, 10)

    st.sidebar.divider()
    db_criteria = [col for cond, col in zip([st.sidebar.checkbox("Subcontractor (Subc)", value=True), st.sidebar.checkbox("Welder ID (Welder1)", value=True), st.sidebar.checkbox("Material Type", value=True), st.sidebar.checkbox("Welding Process", value=True), st.sidebar.checkbox("Line ID", value=False)], ['Subc', 'Welder1', 'MaterialType', 'WPS.1.Description', 'Line']) if cond]

    df_to_process = df_raw.copy()
    if selected_sub != "ALL": df_to_process = df_raw[df_raw['Subc'] == selected_sub].copy()

    engine = RTOptimizerEngine(fallback_perc/100, days_per_lot, db_criteria, location_scope)
    audit_df, df_with_lots = engine.get_lot_audit(df_to_process)

    if audit_df.empty:
        st.warning("No data found.")
    else:
        tab1, tab2 = st.tabs([":material/assignment: Work Order", ":material/dashboard: Dashboard"])
        with tab1:
            if st.button("🚀 Generate Optimized Plan"):
                result = engine.execute_optimization(df_with_lots, audit_df.copy())
                if not result.empty:
                    result['Plan_Date'] = datetime.now().strftime('%d/%m/%Y')
                    display_cols = ['Joint_ID', 'Inspection_Reason', 'Lot_ID', 'Welder1', 'Line', 'MaterialType', 'WPS.1.Description', 'Dateofweld']
                    st.dataframe(result[[c for c in display_cols if c in result.columns]], use_container_width=True, hide_index=True)
                    st.download_button("📥 Download Plan", result.to_csv(sep=';', index=False).encode('utf-8-sig'), f"plan_{selected_sub}.csv", "text/csv")
                else: st.success("✅ Compliance achieved.")

        with tab2:
            st.subheader(f"Dashboard: {selected_sub}")
            k1, k2, k3, k4, k5 = st.columns(5)
            total_l = len(audit_df); open_l = len(audit_df[audit_df['Status'] == '🔴 OPEN'])
            k1.metric("Total Lots", total_l); k2.metric("Open Lots", open_l)
            k3.metric("Lot Compliance", f"{((total_l-open_l)/total_l)*100:.1f}%" if total_l > 0 else "0%")
            k4.metric("Avg. Actual RT %", f"{audit_df['Current_RT_Done_%'].mean():.1f}%" if total_l > 0 else "0%")
            k5.metric("Avg. Target RT %", f"{audit_df['Current_RT_Req'].mean():.1f}%" if total_l > 0 else "0%")
            
            st.divider()
            f1, f2, f3, f4 = st.columns(4)
            with f1: s_lid = st.multiselect("Filter Lot ID", options=sorted(audit_df['Lot_ID'].unique()))
            with f2: s_w = st.multiselect("Filter Welder", options=sorted(audit_df['Welder'].unique()))
            with f3: s_m = st.multiselect("Filter Material", options=sorted(audit_df['Material'].unique()))
            with f4: s_s = st.multiselect("Filter Status", options=['🔴 OPEN', '🟢 CLOSED'])
            f_audit = audit_df.copy()
            if s_lid: f_audit = f_audit[f_audit['Lot_ID'].isin(s_lid)]
            if s_w: f_audit = f_audit[f_audit['Welder'].isin(s_w)]
            if s_m: f_audit = f_audit[f_audit['Material'].isin(s_m)]
            if s_s: f_audit = f_audit[f_audit['Status'].isin(s_s)]

            event = st.dataframe(f_audit, use_container_width=True, on_select="rerun", selection_mode="single-row", hide_index=True)
            if event.selection.rows:
                row_idx = event.selection.rows[0]; lot_id = f_audit.iloc[row_idx]['Lot_ID']
                st.markdown(f"### 🔍 Detailed Explorer: Lot `{lot_id}`")
                # --- AQUÍ SE INCLUYE RT2rej EN EL DETALLE ---
                detail_cols = ['Joint_ID', 'Inspection_Type', 'Line', 'Dateofweld', 'RTDate1', 'RT1rej', 'RT2Date1', 'RT2rej', 'RT_Perc']
                st.dataframe(df_with_lots[df_with_lots['Lot_ID'] == lot_id][detail_cols], use_container_width=True, hide_index=True)
else:
    st.info("💡 Please upload your SQL CSV extraction.")