import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sklearn.compose
import sklearn.impute
from datetime import datetime

# --- 1. PARCHES DE COMPATIBILIDAD IA ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

if not hasattr(sklearn.impute.SimpleImputer, '_fill_dtype'):
    setattr(sklearn.impute.SimpleImputer, '_fill_dtype', float)

# --- 2. CONFIGURACIÓN DE COLUMNAS ---
DB_MAP = {
    'Joint_ID': 'Joint_ID', 'Subc': 'Subc', 'Welder1': 'Welder1', 'Line': 'Line',
    'location': 'location', 'MaterialType': 'MaterialType', 'WPS.1.Description': 'WPS.1.Description',
    'Dateofweld': 'Dateofweld', 'RTDate1': 'RTDate1', 'RT_Perc': 'RT_Perc',
    'RT1rej': 'RT1rej', 'RTAccepted': 'RTAccepted', 'Jointsize': 'Jointsize', 'Thickness': 'Thickness'
}

# --- 3. FUNCIONES DE APOYO ---
@st.cache_resource
def load_ai_model(path):
    if os.path.exists(path):
        try: return joblib.load(path)
        except: return None
    return None

@st.cache_data
def load_and_preprocess_data(file):
    d = pd.read_csv(file, sep=';', encoding='utf-8-sig')
    d.columns = d.columns.str.strip()
    new_cols = {col: DB_MAP[req] for col in d.columns for req in DB_MAP if col.lower() == req.lower()}
    d.rename(columns=new_cols, inplace=True)
    if 'Joint_ID' not in d.columns: d['Joint_ID'] = range(1, len(d) + 1)
    for col in d.select_dtypes(['object']).columns: d[col] = d[col].astype(str).str.strip()
    d['Dateofweld'] = pd.to_datetime(d['Dateofweld'], dayfirst=True, errors='coerce')
    d = d.dropna(subset=['Dateofweld']).copy()
    if 'MaterialType' in d.columns: d = d[d['MaterialType'].str.upper() != 'PLASTIC'].copy()
    return d

# --- 4. OPTIMIZATION ENGINE ---
class RTOptimizerEngine:
    def __init__(self, fallback_rt_perc, window_days, lot_criteria, selected_location_scope, model):
        self.fallback_rt_perc = fallback_rt_perc
        self.window_days = window_days
        self.lot_criteria = lot_criteria 
        self.scope = selected_location_scope
        self.model_pipeline = model

    def _assign_dynamic_blocks(self, df):
        if df.empty: return df
        df = df.sort_values('Dateofweld')
        groups = df.groupby(self.lot_criteria)
        processed_chunks = []
        for _, group in groups:
            group = group.copy()
            start_date = group['Dateofweld'].iloc[0]
            current_block = 0
            block_ids = []
            for date in group['Dateofweld']:
                if pd.isna(date): block_ids.append(-1)
                elif date < start_date + pd.Timedelta(days=self.window_days): block_ids.append(current_block)
                else:
                    start_date = date
                    current_block += 1
                    block_ids.append(current_block)
            group['Block_ID'] = block_ids
            processed_chunks.append(group)
        return pd.concat(processed_chunks)

    def get_lot_audit(self, df):
        d = df.copy()
        d['RTDate1'] = pd.to_datetime(d['RTDate1'], dayfirst=True, errors='coerce')
        for col in ['Jointsize', 'Thickness']:
            if col in d.columns:
                d[col] = d[col].astype(str).str.replace(',', '.')
                d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0)
        d['RT_Perc'] = pd.to_numeric(d['RT_Perc'], errors='coerce').replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        for col in ['RT1rej', 'RTAccepted']:
            if col in d.columns:
                d[col] = d[col].astype(str).str.upper().str.strip()
                d[col] = d[col].map({'TRUE': True, 'FALSE': False, '1': True, '0': False, 'NAN': False}).fillna(False)
            else: d[col] = False

        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        if self.scope == "ALL": df_filtered = d[d['RT_Perc'] < 100].copy()
        else:
            allowed = location_map.get(self.scope, [])
            df_filtered = d[(d['location'].isin(allowed)) & (d['RT_Perc'] < 100)].copy()
        
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame()

        if self.model_pipeline:
            try:
                X_feats = ['Jointsize', 'Subc', 'MaterialType', 'WPS.1.Description', 'Thickness']
                df_filtered['AI_Prob'] = self.model_pipeline.predict_proba(df_filtered[X_feats])[:, 1]
            except: df_filtered['AI_Prob'] = 0.0
        else: df_filtered['AI_Prob'] = 0.0

        def risk_tag(p):
            pct = p * 100
            if p > 0.75: return f"🔴 High ({pct:.1f}%)"
            if p > 0.30: return f"🟡 Med ({pct:.1f}%)"
            return f"🟢 Low ({pct:.1f}%)"
        df_filtered['Risk_Level'] = df_filtered['AI_Prob'].apply(risk_tag)

        df_wb = self._assign_dynamic_blocks(df_filtered)
        def build_id(r):
            parts = [str(r['Block_ID'])]
            for c in self.lot_criteria: parts.append(str(r[c]))
            return "_".join(parts)
        df_wb['Lot_ID'] = df_wb.apply(build_id, axis=1)

        # --- LÓGICA DE PENALTY CRONOLÓGICA ---
        def process_lot_integrity(group):
            group = group.sort_values(['RTDate1', 'Dateofweld'], ascending=[True, True])
            
            types = []; stats = []
            first_fail_idx = -1
            penalty_count = 0
            full_audit_active = False
            
            # Pase 1: Identificar escalada
            for i, (_, row) in enumerate(group.iterrows()):
                # Determinar Status (Puntos 9, 10, 11 Checklist)
                if pd.isna(row['RTDate1']): s = "Not Inspected"
                elif not row['RT1rej']: s = "Standard RT"
                elif row['RTAccepted']: s = "Rejected & Repaired"
                else: s = "Rejected & Pending to be repaired"
                stats.append(s)
                
                # Lógica de Penaltis
                if not full_audit_active:
                    if row['RT1rej'] and first_fail_idx == -1:
                        first_fail_idx = i
                        types.append("Random Inspection Joint")
                    elif first_fail_idx != -1 and penalty_count < 2:
                        types.append("Penalty Tracer")
                        penalty_count += 1
                        if row['RT1rej']: full_audit_active = True
                    else:
                        types.append("Random Inspection Joint")
                else:
                    types.append("Penalty Lot 100%")
            
            group['Inspection_Type'] = types
            group['Inspection_Status'] = stats
            return group

        df_wb = df_wb.groupby('Lot_ID', group_keys=False).apply(process_lot_integrity)
        df_wb['Lot_ID_Col'] = df_wb['Lot_ID'] # Duplicamos para asegurar acceso

        # Auditoría para KPIs
        audit = df_wb.groupby('Lot_ID').agg(
            Total_Joints=('Joint_ID', 'count'), RT1_Count=('RTDate1', 'count'), 
            Rej_Count=('RT1rej', 'sum'), Pending_Repairs=('RTAccepted', lambda x: (x == False).sum()),
            RT_Req=('RT_Perc', 'max'), Welder=('Welder1', 'first'), 
            Subcontractor=('Subc', 'first'), Material=('MaterialType', 'first')
        ).reset_index()
        
        def calc_required(r):
            # Determinamos si el lote escaló a 100% buscando si hay tracers fallidos
            lot_rows = df_wb[df_wb['Lot_ID'] == r['Lot_ID']]
            if any(lot_rows[lot_rows['Inspection_Type'] == "Penalty Tracer"]['RT1rej']):
                return r['Total_Joints']
            base = np.ceil(r['Total_Joints'] * (r['RT_Req'] / 100))
            return int(min(r['Total_Joints'], base + 2 if r['Rej_Count'] >= 1 else base))

        audit['Required'] = audit.apply(calc_required, axis=1)
        audit['Done_%'] = (audit['RT1_Count'] / audit['Total_Joints'] * 100).round(1)
        audit['Deficit'] = (audit['Required'] - audit['RT1_Count']).clip(lower=0)
        audit['Status'] = np.where((audit['Deficit'] == 0) & (audit['Pending_Repairs'] == 0), '🟢 CLOSED', '🔴 OPEN')
        
        return audit, df_wb

    def execute_optimization(self, df_audit_base, audit):
        if audit.empty: return pd.DataFrame()
        debts = audit.set_index('Lot_ID')['Deficit'].to_dict()
        candidates = df_audit_base[df_audit_base['RTDate1'].isna()].copy()
        candidates = candidates.sort_values('AI_Prob', ascending=False)
        
        plan = []
        for _, row in candidates.iterrows():
            l_id = row['Lot_ID']
            if debts.get(l_id, 0) > 0:
                debts[l_id] -= 1
                row['Reason'] = row['Inspection_Type']
                plan.append(row)
        return pd.DataFrame(plan)

# --- UI UTILS ---
def get_dynamic_scopes(df, sub):
    m = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
    locs = df['location'].unique() if sub == "ALL" else df[df['Subc'] == sub]['location'].unique()
    available = [s for s, v in m.items() if any(l in locs for l in v)]
    return ["ALL"] + available if len(available) > 1 else available

# --- UI ---
st.set_page_config(page_title="RT Optimizer v3.1", layout="wide", page_icon="🏗️")
st.title(":material/engineering: RT Optimizer")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_welding_lgb.joblib')
loaded_model = load_ai_model(MODEL_PATH)
if loaded_model: st.sidebar.success("✅ AI Engine Active (LGBM)")
else: st.sidebar.warning("⚠️ Running in Standard Mode")

uploaded_file = st.file_uploader("Upload Daily SQL Extraction", type="csv")

if uploaded_file:
    df_raw = load_and_preprocess_data(uploaded_file)
    subs_list = ["ALL"] + sorted(df_raw['Subc'].unique().tolist())
    selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
    location_scope = st.sidebar.radio("Location Scope:", options=get_dynamic_scopes(df_raw, selected_sub), index=0)
    days_per_lot = st.sidebar.number_input("Days per Window", min_value=1, value=14)
    fallback_perc = st.sidebar.slider("Fallback RT %", 0, 100, 10)

    st.sidebar.divider()
    db_criteria = [col for cond, col in zip([st.sidebar.checkbox("Subcontractor", value=True), st.sidebar.checkbox("Welder", value=True), st.sidebar.checkbox("Material", value=True), st.sidebar.checkbox("Process", value=True), st.sidebar.checkbox("Line ID", value=False)], ['Subc', 'Welder1', 'MaterialType', 'WPS.1.Description', 'Line']) if cond]

    engine = RTOptimizerEngine(fallback_perc/100, days_per_lot, db_criteria, location_scope, loaded_model)
    df_to_proc = df_raw.copy() if selected_sub == "ALL" else df_raw[df_raw['Subc'] == selected_sub].copy()
    audit_df, df_with_lots = engine.get_lot_audit(df_to_proc)

    if audit_df.empty: st.warning("No sampling data found.")
    else:
        tab1, tab2 = st.tabs([":material/assignment: Work Order", ":material/dashboard: Dashboard"])
        with tab1:
            if st.button("🚀 Generate Plan"):
                result = engine.execute_optimization(df_with_lots, audit_df.copy())
                if not result.empty:
                    result['Plan_Date'] = datetime.now().strftime('%d/%m/%Y')
                    st.dataframe(result[['Joint_ID', 'Risk_Level', 'Reason', 'Lot_ID', 'Welder1', 'Line', 'MaterialType']], use_container_width=True, hide_index=True)
                    st.download_button("📥 Download Plan", result.to_csv(sep=';', index=False).encode('utf-8-sig'), f"plan_{selected_sub}.csv", "text/csv")
                else: st.success("✅ Compliance achieved.")

        with tab2:
            k1, k2, k3, k4, k5 = st.columns(5)
            total_l = len(audit_df); open_l = len(audit_df[audit_df['Status'] == '🔴 OPEN'])
            k1.metric("Total Lots", total_l); k2.metric("Open Lots", open_l)
            k3.metric("Project Compliance", f"{((total_l-open_l)/total_l)*100:.1f}%")
            k4.metric("Avg. Actual RT %", f"{audit_df['Done_%'].mean():.1f}%")
            k5.metric("Avg. Target RT %", f"{audit_df['RT_Req'].mean():.1f}%")
            
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
                det_cols = ['Joint_ID', 'Risk_Level', 'Inspection_Type', 'Inspection_Status', 'Line', 'Dateofweld', 'RTDate1', 'RT1rej', 'RTAccepted']
                # USAMOS Lot_ID_Col para evitar el KeyError
                st.dataframe(df_with_lots[df_with_lots['Lot_ID_Col'] == lot_id][det_cols], use_container_width=True, hide_index=True)
                csv_lot = df_with_lots[df_with_lots['Lot_ID_Col'] == lot_id].to_csv(sep=';', index=False).encode('utf-8-sig')
                st.download_button("📥 Download Lot Detail", csv_lot, f"detail_{lot_id}.csv", "text/csv")
else: st.info("💡 Please upload your SQL CSV extraction.")