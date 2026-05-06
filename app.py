import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sklearn.compose
import sklearn.impute
from datetime import datetime

# --- 1. PARCHES DE COMPATIBILIDAD ---
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
    'location': 'location', 'MaterialType': 'MaterialType', 'WPS.1.Description': 'Process',
    'Dateofweld': 'Dateofweld', 'RTDate1': 'RTDate1', 'RT_Perc': 'RT_Perc',
    'RT1rej': 'RT1rej', 'RTAccepted': 'RTAccepted', 'Jointsize': 'Jointsize', 'Thickness': 'Thickness'
}

# --- 3. CARGA DE DATOS ---
@st.cache_resource
def load_ai_model(path):
    if os.path.exists(path):
        try: return joblib.load(path)
        except: return None
    return None

@st.cache_data
def load_and_preprocess(file):
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

# --- 4. ENGINE DE OPTIMIZACIÓN ---
class RTOptimizerEngine:
    def __init__(self, fallback_rt_perc, window_days, lot_criteria, selected_location_scope, model):
        self.fallback_rt_perc = fallback_rt_perc
        self.window_days = window_days
        self.lot_criteria = lot_criteria 
        self.scope = selected_location_scope
        self.model_pipeline = model

    def run_full_process(self, df):
        # A. Saneamiento Numérico y Booleano
        d = df.copy()
        d['RTDate1'] = pd.to_datetime(d['RTDate1'], dayfirst=True, errors='coerce')
        for col in ['Jointsize', 'Thickness']:
            if col in d.columns:
                d[col] = d[col].astype(str).str.replace(',', '.')
                d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0)
        d['RT_Perc'] = pd.to_numeric(d['RT_Perc'], errors='coerce').replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        for col in ['RT1rej', 'RTAccepted']:
            d[col] = d[col].astype(str).str.upper().str.strip().map({'TRUE': True, 'FALSE': False, '1': True, '0': False, 'NAN': False}).fillna(False)

        # B. Scope Filter
        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        if self.scope == "ALL": d_f = d[d['RT_Perc'] < 100].copy()
        else:
            allowed = location_map.get(self.scope, [])
            d_f = d[(d['location'].isin(allowed)) & (d['RT_Perc'] < 100)].copy()
        
        if d_f.empty: return pd.DataFrame(), pd.DataFrame()

        # C. IA
        if self.model_pipeline:
            try:
                X = d_f[['Jointsize', 'Subc', 'MaterialType', 'Process', 'Thickness']]
                d_f['AI_Prob'] = self.model_pipeline.predict_proba(X)[:, 1]
            except: d_f['AI_Prob'] = 0.0
        else: d_f['AI_Prob'] = 0.0
        
        d_f['Risk_Level'] = d_f['AI_Prob'].apply(lambda p: f"🔴 High ({p*100:.1f}%)" if p > 0.75 else (f"🟡 Med ({p*100:.1f}%)" if p > 0.3 else f"🟢 Low ({p*100:.1f}%)"))

        # D. Bloques y Lot_ID
        d_f = d_f.sort_values(['Subc', 'Dateofweld'])
        processed = []
        for _, group in d_f.groupby(self.lot_criteria, dropna=False):
            group = group.copy()
            start_date = group['Dateofweld'].iloc[0]
            current_block = 0
            b_ids = []
            for date in group['Dateofweld']:
                if date >= start_date + pd.Timedelta(days=self.window_days):
                    start_date = date
                    current_block += 1
                b_ids.append(current_block)
            group['Block_ID'] = b_ids
            # Construcción segura de ID
            group['Lot_ID'] = group['Block_ID'].astype(str)
            for c in self.lot_criteria: group['Lot_ID'] += "_" + group[c].astype(str)
            processed.append(group)
        d_wb = pd.concat(processed)

        # E. Auditoría y Lógica Penalty (Mapeo Lineal)
        # 1. Contamos rechazos por Lote
        rej_counts = d_wb.groupby('Lot_ID')['RT1rej'].sum().to_dict()
        
        # 2. Etiquetamos cada junta (Puntos 9-12 Checklist)
        def label_rows(group):
            group = group.sort_values(['RTDate1', 'Dateofweld'], ascending=[True, True])
            n_rej = rej_counts.get(group.name, 0)
            
            types = []; stats = []
            fail_found = False; tracers = 0
            for _, row in group.iterrows():
                # Status
                if pd.isna(row['RTDate1']): stats.append("Not Inspected")
                elif not row['RT1rej']: stats.append("Standard RT")
                elif row['RTAccepted']: stats.append("Rejected & Repaired")
                else: stats.append("Rejected & Pending to be repaired")
                # Type
                if n_rej > 1: types.append("Penalty Lot 100%")
                elif n_rej == 1:
                    if row['RT1rej'] and not fail_found:
                        types.append("Random Inspection Joint"); fail_found = True
                    elif fail_found and tracers < 2:
                        types.append("Penalty Tracer"); tracers += 1
                        if row['RT1rej']: # Si el tracer falla, activamos 100% para el resto
                            types[-1] = "Penalty Lot 100%"
                            n_rej = 2 
                    else: types.append("Random Inspection Joint")
                else: types.append("Random Inspection Joint")
            group['Inspection_Type'] = types
            group['Inspection_Status'] = stats
            return group

        df_final = d_wb.groupby('Lot_ID', group_keys=False).apply(label_rows)

        # 3. Crear tabla resumen (Log)
        audit = df_final.groupby('Lot_ID').agg(
            Total_Joints=('Joint_ID', 'count'), RT1_Count=('RTDate1', 'count'), 
            Rej_Count=('RT1rej', 'sum'), Not_Accepted=('RTAccepted', lambda x: (x == False).sum()),
            RT_Req=('RT_Perc', 'max'), Welder=('Welder1', 'first'), 
            Subcontractor=('Subc', 'first'), Material=('MaterialType', 'first')
        ).reset_index()

        # Requisito final basado en etiquetas reales
        def get_req(row):
            lot_data = df_final[df_final['Lot_ID'] == row['Lot_ID']]
            if "Penalty Lot 100%" in lot_data['Inspection_Type'].values: return row['Total_Joints']
            base = np.ceil(row['Total_Joints'] * (row['RT_Req'] / 100))
            return int(min(row['Total_Joints'], base + 2 if row['Rej_Count'] >= 1 else base))

        audit['Required'] = audit.apply(get_req, axis=1)
        audit['Deficit'] = (audit['Required'] - audit['RT1_Count']).clip(lower=0)
        audit['Done_%'] = (audit['RT1_Count'] / audit['Total_Joints'] * 100).round(1)
        audit['Status'] = np.where((audit['Deficit'] == 0) & (audit['Not_Accepted'] == 0), '🟢 CLOSED', '🔴 OPEN')
        
        return audit, df_final

    def execute_optimization(self, df_with_lots, audit):
        if audit.empty: return pd.DataFrame()
        debts = audit.set_index('Lot_ID')['Deficit'].to_dict()
        candidates = df_with_lots[df_with_lots['RTDate1'].isna()].sort_values('AI_Prob', ascending=False)
        plan = []
        for _, row in candidates.iterrows():
            l_id = row['Lot_ID']
            if debts.get(l_id, 0) > 0:
                debts[l_id] -= 1
                row['Reason'] = row['Inspection_Type']
                plan.append(row)
        return pd.DataFrame(plan)

# --- UI ---
st.set_page_config(page_title="RT Optimizer 4.0", layout="wide", page_icon="🏗️")
st.title(":material/engineering: RT Optimizer")

# Carga ML (Punto 3)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_welding_lgb.joblib')
loaded_model = load_ai_model(MODEL_PATH)
if loaded_model: st.sidebar.success("✅ AI Engine Active (LGBM)")
else: st.sidebar.warning("⚠️ Standard Mode Active")

uploaded_file = st.file_uploader("Upload Daily SQL Extraction", type="csv")

if uploaded_file:
    df_raw = load_and_preprocess(uploaded_file)
    # --- SELECTORES DINÁMICOS (Punto 1) ---
    subs_list = ["ALL"] + sorted(df_raw['Subc'].unique().tolist())
    selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
    
    def get_scopes(df, sub):
        m = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        locs = df['location'].unique() if sub == "ALL" else df[df['Subc'] == sub]['location'].unique()
        return ["ALL"] + [s for s, v in m.items() if any(l in locs for l in v)]

    location_scope = st.sidebar.radio("Location Scope:", options=get_scopes(df_raw, selected_sub), index=0)
    days_per_lot = st.sidebar.number_input("Days per Window", min_value=1, value=14)
    fallback_perc = st.sidebar.slider("Fallback RT %", 0, 100, 10)

    st.sidebar.divider()
    # Punto 2
    c_sub = st.sidebar.checkbox("Subcontractor (Subc)", value=True)
    c_wel = st.sidebar.checkbox("Welder ID (Welder1)", value=True)
    c_mat = st.sidebar.checkbox("Material Type", value=True)
    c_pro = st.sidebar.checkbox("Welding Process", value=True)
    c_lin = st.sidebar.checkbox("Line ID", value=False)
    criteria = [col for cond, col in zip([c_sub, c_wel, c_mat, c_pro, c_lin], ['Subc', 'Welder1', 'MaterialType', 'Process', 'Line']) if cond]

    # --- EJECUCIÓN ---
    engine = RTOptimizerEngine(fallback_perc/100, days_per_lot, criteria, location_scope, loaded_model)
    df_input = df_raw.copy() if selected_sub == "ALL" else df_raw[df_raw['Subc'] == selected_sub].copy()
    audit_df, df_with_lots = engine.run_full_process(df_input)

    if audit_df.empty: st.warning("No sampling data found.")
    else:
        t1, t2 = st.tabs([":material/assignment: Work Order", ":material/dashboard: Dashboard"])
        with t1:
            if st.button("🚀 Generate Optimized Plan"):
                res = engine.execute_optimization(df_with_lots, audit_df.copy())
                if not res.empty:
                    res['Plan_Date'] = datetime.now().strftime('%d/%m/%Y')
                    st.dataframe(res[['Joint_ID', 'Risk_Level', 'Reason', 'Lot_ID', 'Welder1', 'Line', 'MaterialType']], use_container_width=True, hide_index=True)
                    st.download_button("📥 Download Plan", res.to_csv(sep=';', index=False).encode('utf-8-sig'), f"plan_{selected_sub}.csv")
                else: st.success("✅ Compliance achieved.")

        with t2:
            # Punto 5
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Lots", len(audit_df)); k2.metric("Open Lots", len(audit_df[audit_df['Status'] == '🔴 OPEN']))
            k3.metric("Compliance", f"{(len(audit_df[audit_df['Status'] == '🟢 CLOSED'])/len(audit_df)*100 if len(audit_df)>0 else 0):.1f}%")
            k4.metric("Avg. Actual RT %", f"{audit_df['Done_%'].mean():.1f}%"); k5.metric("Avg. Target RT %", f"{audit_df['RT_Req'].mean():.1f}%")
            st.divider()
            f1, f2, f3, f4 = st.columns(4)
            with f1: sl = st.multiselect("Lot ID", options=sorted(audit_df['Lot_ID'].unique()))
            with f2: sw = st.multiselect("Welder", options=sorted(audit_df['Welder'].unique()))
            with f3: sm = st.multiselect("Material", options=sorted(audit_df['Material'].unique()))
            with f4: ss = st.multiselect("Status", options=['🔴 OPEN', '🟢 CLOSED'])
            f_audit = audit_df.copy()
            if sl: f_audit = f_audit[f_audit['Lot_ID'].isin(sl)]
            if sw: f_audit = f_audit[f_audit['Welder'].isin(sw)]
            if sm: f_audit = f_audit[f_audit['Material'].isin(sm)]
            if ss: f_audit = f_audit[f_audit['Status'].isin(ss)]

            st.subheader("All Lots Summary Log")
            event = st.dataframe(f_audit, use_container_width=True, on_select="rerun", selection_mode="single-row", hide_index=True)
            if event.selection.rows:
                row_idx = event.selection.rows[0]; lid = f_audit.iloc[row_idx]['Lot_ID']
                st.markdown(f"### 🔍 Detailed Explorer: Lot `{lid}`")
                # Punto 6, 9-12
                det_cols = ['Joint_ID', 'Risk_Level', 'Inspection_Type', 'Inspection_Status', 'Line', 'Dateofweld', 'RTDate1', 'RT1rej', 'RTAccepted']
                st.dataframe(df_with_lots[df_with_lots['Lot_ID'] == lid][det_cols], use_container_width=True, hide_index=True)
                st.download_button("📥 Download Lot Detail", df_with_lots[df_with_lots['Lot_ID'] == lid].to_csv(sep=';', index=False).encode('utf-8-sig'), f"detail_{lid}.csv")
else: st.info("💡 Please upload your SQL CSV extraction.")