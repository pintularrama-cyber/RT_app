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
COL_SUBC = 'Subc'
COL_WELDER = 'Welder1'
COL_LINE = 'Line'
COL_LOC = 'location'
COL_MAT = 'MaterialType'
COL_WPS = 'WPS.1.Description'
COL_DATE = 'Dateofweld'
COL_RT1_DATE = 'RTDate1'
COL_PERC = 'RT_Perc'
COL_RT1_REJ = 'RT1rej'
COL_ACCEPTED = 'RTAccepted'
COL_SIZE = 'Jointsize'
COL_THK = 'Thickness'

REQUIRED_INTERNAL = [COL_SUBC, COL_WELDER, COL_LINE, COL_LOC, COL_MAT, COL_WPS, COL_DATE, COL_RT1_DATE, COL_PERC, COL_RT1_REJ, COL_ACCEPTED, COL_SIZE, COL_THK]

# --- 3. FUNCIONES DE CARGA Y SANEAMIENTO ---
@st.cache_resource
def load_ai_model(path):
    if os.path.exists(path):
        try: return joblib.load(path)
        except: return None
    return None

@st.cache_data
def load_and_normalize_data(file):
    # Definimos valores que deben tratarse como vacíos para que no falseen el conteo
    d = pd.read_csv(file, sep=';', encoding='utf-8-sig', na_values=[' ', '', 'None', 'nan', 'NAN'])
    d.columns = d.columns.str.strip()
    
    mapping = {c.lower(): c for c in d.columns}
    new_cols = {mapping[req.lower()]: req for req in REQUIRED_INTERNAL if req.lower() in mapping}
    d.rename(columns=new_cols, inplace=True)
    
    if 'Joint_ID' not in d.columns: d['Joint_ID'] = range(1, len(d) + 1)
    
    for col in d.select_dtypes(['object']).columns: d[col] = d[col].astype(str).str.strip()
    
    d[COL_DATE] = pd.to_datetime(d[COL_DATE], errors='coerce')
    d = d.dropna(subset=[COL_DATE]).copy()
    if COL_MAT in d.columns:
        d = d[d[COL_MAT].str.upper() != 'PLASTIC'].copy()
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
        df = df.sort_values(COL_DATE)
        groups = df.groupby(self.lot_criteria)
        processed_chunks = []
        for _, group in groups:
            group = group.copy()
            start_date = group[COL_DATE].iloc[0]
            current_block = 0
            block_ids = []
            for date in group[COL_DATE]:
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
        # Forzamos que los vacíos sean NaT reales
        d[COL_RT1_DATE] = pd.to_datetime(d[COL_RT1_DATE], errors='coerce')
        
        for col in [COL_SIZE, COL_THK]:
            if col in d.columns:
                d[col] = d[col].astype(str).str.replace(',', '.')
                d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0)
        
        d[COL_PERC] = pd.to_numeric(d[COL_PERC], errors='coerce').replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        
        for col in [COL_RT1_REJ, COL_ACCEPTED]:
            if col in d.columns:
                d[col] = d[col].astype(str).str.upper()
                d[col] = d[col].map({'TRUE': True, 'FALSE': False, '1': True, '0': False, 'NAN': False}).fillna(False)
            else: d[col] = False

        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        if self.scope == "ALL":
            df_filtered = d[d[COL_PERC] < 100].copy()
        else:
            allowed = location_map.get(self.scope, [])
            df_filtered = d[(d[COL_LOC].isin(allowed)) & (d[COL_PERC] < 100)].copy()
        
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame()

        if self.model_pipeline:
            try:
                X = df_filtered[[COL_SIZE, COL_SUBC, COL_MAT, COL_WPS, COL_THK]]
                df_filtered['AI_Prob'] = self.model_pipeline.predict_proba(X)[:, 1]
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

        # Auditoría corregida
        audit = df_wb.groupby('Lot_ID').agg(
            Total_Joints=('Joint_ID', 'count'), 
            RT1_Count=(COL_RT1_DATE, 'count'), # Cuenta no nulos reales
            RT1_Rej=(COL_RT1_REJ, 'sum'),
            Not_Accepted=(COL_ACCEPTED, lambda x: (x == False).sum()),
            RT_Req=(COL_PERC, 'max'), Welder=(COL_WELDER, 'first'), Subcontractor=(COL_SUBC, 'first')
        ).reset_index()
        
        audit['Current_Done'] = audit['RT1_Count'] - audit['RT1_Rej']
        
        def calc_req(r):
            base = np.ceil(r['Total_Joints'] * (r['RT_Req'] / 100))
            if r['RT1_Rej'] == 1: return min(r['Total_Joints'], base + 2)
            if r['RT1_Rej'] > 1: return r['Total_Joints']
            return base

        audit['Required'] = audit.apply(calc_req, axis=1).astype(int)
        audit['Done_%'] = (audit['Current_Done'] / audit['Total_Joints'] * 100).round(1)
        audit['Deficit'] = (audit['Required'] - audit['Current_Done']).clip(lower=0)
        audit['Status'] = np.where((audit['Deficit'] == 0) & (audit['Not_Accepted'] == 0), '🟢 CLOSED', '🔴 OPEN')
        
        # Etiquetado para detalle
        lot_rej_map = audit.set_index('Lot_ID')['RT1_Rej'].to_dict()
        def label_log(row):
            n_rej = lot_rej_map.get(row['Lot_ID'], 0)
            if row[COL_RT1_REJ] and row[COL_ACCEPTED]: return "Rejected & Repaired"
            if row[COL_RT1_REJ] and not row[COL_ACCEPTED]: return "Rejected & pending reparation"
            if pd.notna(row[COL_RT1_DATE]): return "Standard OK"
            if n_rej == 1: return "PENALTY TRACER"
            if n_rej > 1: return "PENALTY LOT"
            return "Standard Sampling"
        
        df_wb['Inspection_Type'] = df_wb.apply(label_log, axis=1)
        return audit, df_wb

    def execute_optimization(self, df_audit_base, audit):
        if audit.empty: return pd.DataFrame()
        debts = audit.set_index('Lot_ID')['Deficit'].to_dict()
        penalty_status = audit.set_index('Lot_ID')['RT1_Rej'].to_dict()
        penalty_assigned = {l_id: 0 for l_id in audit['Lot_ID']}
        
        # Candidatas: No tienen fecha de RT1
        candidates = df_audit_base[df_audit_base[COL_RT1_DATE].isna()].copy()
        
        plan = []
        while sum(debts.values()) > 0 and not candidates.empty:
            def calc_impact(row):
                l_id = row['Lot_ID']
                impact = 1.0 if debts.get(l_id, 0) > 0 else 0.0
                impact += row.get('AI_Prob', 0.0)
                return impact
            
            candidates['Impact'] = candidates.apply(calc_impact, axis=1)
            if candidates['Impact'].max() <= 0: break
            
            best_idx = candidates['Impact'].idxmax(); selected = candidates.loc[best_idx].copy()
            l_id = selected['Lot_ID']; n_rej = penalty_status.get(l_id, 0)
            
            if n_rej > 1: reason = "PENALTY LOT"
            elif n_rej == 1 and penalty_assigned[l_id] < 2:
                reason = "PENALTY TRACER"; penalty_assigned[l_id] += 1
            else: reason = "Standard Sampling"
            
            selected['Inspection_Reason'] = reason
            debts[l_id] -= 1
            plan.append(selected); candidates = candidates.drop(best_idx)
            
        return pd.DataFrame(plan)

# --- UI ---
st.set_page_config(page_title="RT Optimizer", layout="wide", page_icon="🏗️")
st.title(":material/engineering: RT Optimizer")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_welding_lgb.joblib')
loaded_model = load_ai_model(MODEL_PATH)

if loaded_model: st.sidebar.success("✅ AI Engine Active (LGBM)")
else: st.sidebar.warning("⚠️ Running in Standard Mode")

uploaded_file = st.file_uploader("Upload Daily SQL Extraction", type="csv")

if uploaded_file:
    df_raw = load_and_normalize_data(uploaded_file)
    subs_list = ["ALL"] + sorted(df_raw[COL_SUBC].unique().tolist())
    selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
    
    def get_avail(df, sub):
        m = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        locs = df[COL_LOC].unique() if sub == "ALL" else df[df[COL_SUBC] == sub][COL_LOC].unique()
        return ["ALL"] + [s for s, v in m.items() if any(l in locs for l in v)]

    location_scope = st.sidebar.radio("Location Scope:", options=get_avail(df_raw, selected_sub), index=0)
    days_per_lot = st.sidebar.number_input("Days per Window", min_value=1, value=14)
    fallback_perc = st.sidebar.slider("Fallback RT %", 0, 100, 10)

    st.sidebar.divider()
    db_criteria = [col for cond, col in zip([st.sidebar.checkbox("Subcontractor (Subc)", value=True), st.sidebar.checkbox("Welder ID (Welder1)", value=True), st.sidebar.checkbox("Material Type", value=True), st.sidebar.checkbox("Welding Process", value=True), st.sidebar.checkbox("Line ID", value=False)], [COL_SUBC, COL_WELDER, COL_MAT, COL_WPS, COL_LINE]) if cond]

    engine = RTOptimizerEngine(fallback_perc/100, days_per_lot, db_criteria, location_scope, loaded_model)
    df_to_proc = df_raw.copy() if selected_sub == "ALL" else df_raw[df_raw[COL_SUBC] == selected_sub].copy()
    audit_df, df_with_lots = engine.get_lot_audit(df_to_proc)

    if audit_df.empty: st.warning("No sampling data found. Check if dates are correct or location matches.")
    else:
        tab1, tab2 = st.tabs([":material/assignment: Work Order", ":material/dashboard: Dashboard"])
        with tab1:
            if st.button("🚀 Generate Optimized Plan"):
                result = engine.execute_optimization(df_with_lots, audit_df.copy())
                if not result.empty:
                    result['Plan_Date'] = datetime.now().strftime('%d/%m/%Y')
                    st.dataframe(result[['Joint_ID', 'Risk_Level', 'Inspection_Reason', 'Lot_ID', COL_WELDER, COL_LINE, COL_MAT, COL_WPS]], use_container_width=True, hide_index=True)
                    st.download_button("📥 Download", result.to_csv(sep=';', index=False).encode('utf-8-sig'), f"plan_{selected_sub}.csv", "text/csv")
                else: 
                    # --- MENSAJE INFORMATIVO PARA Lotes Abiertos por Reparación ---
                    if any(audit_df['Not_Accepted'] > 0):
                        st.info("⚠️ All sampling quotas met, but some lots remain OPEN due to pending reparations (RTAccepted = FALSE). Please check Dashboard.")
                    else:
                        st.success("✅ Compliance achieved. All lots closed.")

        with tab2:
            # Dashboard KPIs y Tabla
            k1, k2, k3, k4, k5 = st.columns(5)
            total_l = len(audit_df); open_l = len(audit_df[audit_df['Status'] == '🔴 OPEN'])
            k1.metric("Total Lots", total_l); k2.metric("Open Lots", open_l)
            k3.metric("Project Compliance", f"{((total_l-open_l)/total_l)*100:.1f}%" if total_l > 0 else "0%")
            k4.metric("Avg. Actual RT %", f"{audit_df['Done_%'].mean():.1f}%")
            k5.metric("Avg. Target RT %", f"{audit_df['RT_Req'].mean():.1f}%")
            st.divider()
            
            f1, f2, f3 = st.columns(3)
            with f1: s_lid = st.multiselect("Filter Lot ID", options=sorted(audit_df['Lot_ID'].unique()))
            with f2: s_w = st.multiselect("Filter Welder", options=sorted(audit_df['Welder'].unique()))
            with f3: s_s = st.multiselect("Filter Status", options=['🔴 OPEN', '🟢 CLOSED'])
            f_audit = audit_df.copy()
            if s_lid: f_audit = f_audit[f_audit['Lot_ID'].isin(s_lid)]
            if s_w: f_audit = f_audit[f_audit['Welder'].isin(s_w)]
            if s_s: f_audit = f_audit[f_audit['Status'].isin(s_s)]

            event = st.dataframe(f_audit, use_container_width=True, on_select="rerun", selection_mode="single-row", hide_index=True)
            if event.selection.rows:
                row_idx = event.selection.rows[0]; lot_id = f_audit.iloc[row_idx]['Lot_ID']
                st.markdown(f"### 🔍 Detailed Explorer: Lot `{lot_id}`")
                st.dataframe(df_with_lots[df_with_lots['Lot_ID'] == lot_id][['Joint_ID', 'Risk_Level', 'Inspection_Type', COL_LINE, COL_DATE, COL_RT1_DATE, COL_RT1_REJ, COL_ACCEPTED, COL_PERC]], use_container_width=True, hide_index=True)