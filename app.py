import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sklearn.compose
import sklearn.impute
from datetime import datetime

# --- PARCHE NINJA DE COMPATIBILIDAD IA ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

if not hasattr(sklearn.impute.SimpleImputer, '_fill_dtype'):
    setattr(sklearn.impute.SimpleImputer, '_fill_dtype', float)

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_ai_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except: return None
    return None

@st.cache_data
def load_and_preprocess_data(file):
    d = pd.read_csv(file, sep=';', encoding='utf-8-sig')
    d.columns = d.columns.str.strip()
    for col in d.select_dtypes(['object']).columns:
        d[col] = d[col].astype(str).str.strip()
    if 'MaterialType' in d.columns:
        d = d[d['MaterialType'].str.upper() != 'PLASTIC'].copy()
    return d

# --- OPTIMIZATION ENGINE ---
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
        d['Dateofweld'] = pd.to_datetime(d['Dateofweld'], dayfirst=True, errors='coerce')
        d['RTDate1'] = pd.to_datetime(d['RTDate1'], dayfirst=True, errors='coerce')
        d['RT2Date1'] = pd.to_datetime(d.get('RT2Date1', None), dayfirst=True, errors='coerce')
        d = d.dropna(subset=['Dateofweld'])
        
        for col in ['Jointsize', 'Thickness']:
            if col in d.columns:
                d[col] = d[col].astype(str).str.replace(',', '.')
                d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0)
        
        d['RT_Perc'] = pd.to_numeric(d['RT_Perc'], errors='coerce').replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        
        if 'RT1rej' not in d.columns: d['RT1rej'] = False
        else:
            d['RT1rej'] = d['RT1rej'].astype(str).str.upper().str.strip()
            d['RT1rej'] = d['RT1rej'].map({'TRUE': True, 'FALSE': False, '1': True, '0': False, 'NAN': False}).fillna(False)

        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        if self.scope == "ALL":
            df_filtered = d[d['RT_Perc'] < 100].copy()
        else:
            allowed_values = location_map.get(self.scope, [])
            df_filtered = d[(d['location'].isin(allowed_values)) & (d['RT_Perc'] < 100)].copy()
        
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame()

        # IA
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

        # Lotes
        df_wb = self._assign_dynamic_blocks(df_filtered)
        def build_id(r):
            parts = [str(r['Block_ID'])]
            for c in self.lot_criteria: parts.append(str(r[c]) if c in r else "NA")
            return "_".join(parts)
        df_wb['Lot_ID'] = df_wb.apply(build_id, axis=1)

        audit = df_wb.groupby('Lot_ID').agg(
            Total_Joints=('Joint_ID', 'count'), RT1_Done=('RTDate1', 'count'), RT1_Rej=('RT1rej', 'sum'),
            RT2_Pend=('RT1rej', lambda x: ((x==True) & (df_wb.loc[x.index, 'RT2Date1'].isna())).sum()),
            RT_Req=('RT_Perc', 'max'), Welder=('Welder1', 'first'), Process=('WPS.1.Description', 'first'),
            Material=('MaterialType', 'first'), Subcontractor=('Subc', 'first'), Block_Start=('Dateofweld', 'min')
        ).reset_index()
        
        audit['Current_Done'] = audit['RT1_Done'] - audit['RT1_Rej']
        audit['Required'] = audit.apply(lambda r: min(r['Total_Joints'], np.ceil(r['Total_Joints']*(r['RT_Req']/100)) + (2 if r['RT1_Rej']==1 else (r['Total_Joints'] if r['RT1_Rej']>1 else 0))), axis=1).astype(int)
        audit['Done_%'] = (audit['Current_Done'] / audit['Total_Joints'] * 100).round(1)
        audit['Deficit'] = (audit['Required'] - audit['Current_Done']).clip(lower=0)
        audit['Status'] = np.where((audit['Deficit'] == 0) & (audit['RT2_Pend'] == 0), '🟢 CLOSED', '🔴 OPEN')
        
        def labeler(row):
            lot_r = audit[audit['Lot_ID'] == row['Lot_ID']].iloc[0]
            if row['RT1rej'] and pd.notna(row['RT2Date1']): return "🛠️ REPAIR DONE"
            if row['RT1rej']: return "❌ REJECTED"
            if pd.notna(row['RTDate1']): return "✅ STANDARD (OK)"
            if lot_r['RT1_Rej'] == 1: return "🚨 PENALTY"
            if lot_r['RT1_Rej'] > 1: return "🧨 FULL AUDIT"
            return "Standard"
        df_wb['Inspection_Type'] = df_wb.apply(labeler, axis=1)

        return audit, df_wb

    def execute_optimization(self, df_audit_base, audit):
        if audit.empty: return pd.DataFrame()
        debts = audit.set_index('Lot_ID')['Deficit'].to_dict()
        penalty_status = audit.set_index('Lot_ID')['RT1_Rej'].to_dict()
        candidates = df_audit_base[df_audit_base['RTDate1'].isnull()].copy()
        plan = []
        while sum(debts.values()) > 0 and not candidates.empty:
            def calc_impact(row):
                l_id = row['Lot_ID']
                impact = 1.0 if debts.get(l_id, 0) > 0 else 0.0
                impact += row.get('AI_Prob', 0.0)
                return impact
            candidates['Impact'] = candidates.apply(calc_impact, axis=1)
            if candidates['Impact'].max() == 0: break
            best_idx = candidates['Impact'].idxmax(); selected = candidates.loc[best_idx].copy()
            l_id = selected['Lot_ID']; n_rej = penalty_status.get(l_id, 0)
            selected['Inspection_Reason'] = "🚨 PENALTY" if n_rej == 1 else ("🧨 FULL AUDIT" if n_rej > 1 else "Standard")
            if debts.get(l_id, 0) > 0: debts[l_id] -= 1
            plan.append(selected); candidates = candidates.drop(best_idx)
        return pd.DataFrame(plan)

# --- UI ---
st.set_page_config(page_title="RT Optimizer", layout="wide", page_icon="🏗️")
st.title(":material/engineering: RT Optimizer")

# CACHED MODEL LOAD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_welding_lgb.joblib')
loaded_model = load_ai_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload SQL Extraction (CSV)", type="csv")

if uploaded_file:
    df_raw = load_and_preprocess_data(uploaded_file)
    required_cols = ['Joint_ID', 'Subc', 'Welder1', 'Line', 'location', 'MaterialType', 'WPS.1.Description', 'Dateofweld', 'RTDate1', 'RT_Perc', 'RT1rej', 'RT2Date1', 'Jointsize', 'Thickness']
    new_cols = {col: req for col in df_raw.columns for req in required_cols if col.lower() == req.lower()}
    df_raw.rename(columns=new_cols, inplace=True)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    subs_list = ["ALL"] + sorted(df_raw['Subc'].unique().tolist())
    selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
    
    def get_avail(df, sub):
        m = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        locs = df['location'].unique() if sub == "ALL" else df[df['Subc'] == sub]['location'].unique()
        available = [s for s, v in m.items() if any(l in locs for l in v)]
        return ["ALL"] + available if len(available) > 1 else available

    scopes = get_avail(df_raw, selected_sub)
    location_scope = st.sidebar.radio("Location Scope:", options=scopes, index=0)
    days_per_lot = st.sidebar.number_input("Days per Window", min_value=1, value=14)
    fallback_perc = st.sidebar.slider("Fallback RT %", 0, 100, 10)

    st.sidebar.divider()
    db_criteria = [col for cond, col in zip([st.sidebar.checkbox("Subcontractor", value=True), st.sidebar.checkbox("Welder", value=True), st.sidebar.checkbox("Material Type", value=True), st.sidebar.checkbox("Welding Process", value=True), st.sidebar.checkbox("Line ID", value=False)], ['Subc', 'Welder1', 'MaterialType', 'WPS.1.Description', 'Line']) if cond]

    engine = RTOptimizerEngine(fallback_perc/100, days_per_lot, db_criteria, location_scope, loaded_model)
    df_to_proc = df_raw.copy() if selected_sub == "ALL" else df_raw[df_raw['Subc'] == selected_sub].copy()
    audit_df, df_with_lots = engine.get_lot_audit(df_to_proc)

    if audit_df.empty: st.warning("No data found.")
    else:
        tab1, tab2 = st.tabs([":material/assignment: Work Order", ":material/dashboard: Dashboard"])
        with tab1:
            if st.button("🚀 Generate Optimized Plan"):
                with st.spinner('Calculating...'):
                    result = engine.execute_optimization(df_with_lots, audit_df.copy())
                    if not result.empty:
                        result['Plan_Date'] = datetime.now().strftime('%d/%m/%Y')
                        st.dataframe(result[['Joint_ID', 'Risk_Level', 'Inspection_Reason', 'Lot_ID', 'Welder1', 'Line', 'MaterialType', 'WPS.1.Description']], use_container_width=True, hide_index=True)
                        st.download_button("📥 Download", result.to_csv(sep=';', index=False).encode('utf-8-sig'), f"plan_{selected_sub}.csv", "text/csv")
                    else: st.success("✅ Compliance achieved.")

        with tab2:
            st.subheader(f"Status: {selected_sub} | {location_scope}")
            k1, k2, k3, k4, k5 = st.columns(5)
            total_l = len(audit_df); open_l = len(audit_df[audit_df['Status'] == '🔴 OPEN'])
            k1.metric("Total Lots", total_l); k2.metric("Open Lots", open_l)
            k3.metric("Project Compliance", f"{((total_l-open_l)/total_l)*100:.1f}%" if total_l > 0 else "0%")
            k4.metric("Avg. Actual RT %", f"{audit_df['Done_%'].mean():.1f}%" if total_l > 0 else "0%")
            k5.metric("Avg. Target RT %", f"{audit_df['RT_Req'].mean():.1f}%" if total_l > 0 else "0%")
            st.divider()
            f1, f2, f3, f4 = st.columns(4)
            with f1: s_lid = st.multiselect("Lot ID", options=sorted(audit_df['Lot_ID'].unique()))
            with f2: s_w = st.multiselect("Welder", options=sorted(audit_df['Welder'].unique()))
            with f3: s_m = st.multiselect("Material", options=sorted(audit_df['Material'].unique()))
            with f4: s_s = st.multiselect("Status", options=['🔴 OPEN', '🟢 CLOSED'])
            f_audit = audit_df.copy()
            if s_lid: f_audit = f_audit[f_audit['Lot_ID'].isin(s_lid)]
            if s_w: f_audit = f_audit[f_audit['Welder'].isin(s_w)]
            if s_m: f_audit = f_audit[f_audit['Material'].isin(s_m)]
            if s_s: f_audit = f_audit[f_audit['Status'].isin(s_s)]
            event = st.dataframe(f_audit, use_container_width=True, on_select="rerun", selection_mode="single-row", hide_index=True)
            if event.selection.rows:
                row_idx = event.selection.rows[0]; lot_id = f_audit.iloc[row_idx]['Lot_ID']
                st.markdown(f"### 🔍 Detailed Explorer: Lot `{lot_id}`")
                det_cols = ['Joint_ID', 'Risk_Level', 'Inspection_Type', 'Line', 'Jointsize', 'Thickness', 'Dateofweld', 'RTDate1', 'RT1rej', 'RT2Date1', 'RT_Perc']
                st.dataframe(df_with_lots[df_with_lots['Lot_ID'] == lot_id][[c for c in det_cols if c in df_with_lots.columns]], use_container_width=True, hide_index=True)
else:
    st.info("💡 Please upload your SQL CSV extraction.")
    schema_df = pd.DataFrame({'Mandatory Column Name': required_cols})
    st.table(schema_df)