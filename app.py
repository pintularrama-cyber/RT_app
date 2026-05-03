import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- OPTIMIZATION ENGINE ---
class RTOptimizerEngine:
    def __init__(self, fallback_rt_perc, window_days, lot_criteria, selected_location_scope, model_path=None):
        self.fallback_rt_perc = fallback_rt_perc
        self.window_days = window_days
        self.lot_criteria = lot_criteria 
        self.scope = selected_location_scope
        
        # CARGA DEL PIPELINE DE IA
        self.model_pipeline = None
        if model_path:
            try:
                # Cargamos el pipeline (que incluye feature engineering + xgboost)
                self.model_pipeline = joblib.load(model_path)
            except Exception as e:
                st.sidebar.error(f"⚠️ Model file not found or invalid: {model_path}")

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
        # Saneamiento fechas
        d['Dateofweld'] = pd.to_datetime(d['Dateofweld'], errors='coerce')
        d['RTDate1'] = pd.to_datetime(d['RTDate1'], errors='coerce')
        d['RT2Date1'] = pd.to_datetime(d.get('RT2Date1', None), errors='coerce')
        d = d.dropna(subset=['Dateofweld'])
        if d.empty: return pd.DataFrame(), pd.DataFrame()
        
        # Saneamiento números
        d['RT_Perc'] = pd.to_numeric(d['RT_Perc'], errors='coerce').replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        d['Jointsize'] = pd.to_numeric(d.get('Jointsize', 0), errors='coerce').fillna(0)
        d['Thickness'] = pd.to_numeric(d.get('Thickness', 0), errors='coerce').fillna(0)
        
        # Saneamiento RT1rej
        if 'RT1rej' not in d.columns: d['RT1rej'] = False
        else:
            d['RT1rej'] = d['RT1rej'].astype(str).str.upper().str.strip()
            d['RT1rej'] = d['RT1rej'].map({'TRUE': True, 'FALSE': False, '1': True, '0': False, 'NAN': False}).fillna(False)

        # Filtro de Ubicación
        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        if self.scope == "ALL":
            df_filtered = d[d['RT_Perc'] < 100].copy()
        else:
            allowed_values = location_map.get(self.scope, [])
            df_filtered = d[(d['location'].isin(allowed_values)) & (d['RT_Perc'] < 100)].copy()
        
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame()

        # --- INFERENCIA NINJA (IA) ---
        if self.model_pipeline:
            try:
                # Preparamos las variables exactas que pide el modelo
                X = df_filtered[['Jointsize', 'Subc', 'MaterialType', 'WPS.1.Description', 'Thickness']]
                # Obtenemos la probabilidad de fallo (clase 1)
                df_filtered['AI_Fail_Prob'] = self.model_pipeline.predict_proba(X)[:, 1]
            except Exception as e:
                st.warning(f"AI Prediction failed: {e}. Using standard sampling.")
                df_filtered['AI_Fail_Prob'] = 0.0
        else:
            df_filtered['AI_Fail_Prob'] = 0.0

        # Semáforo de Riesgo
        def get_risk_label(p):
            if p > 0.75: return "🔴 High"
            if p > 0.30: return "🟡 Medium"
            return "🟢 Low"
        df_filtered['Risk_Level'] = df_filtered['AI_Fail_Prob'].apply(get_risk_label)

        # Asignación de Bloques y Lotes
        df_with_blocks = self._assign_dynamic_blocks(df_filtered)
        
        def build_lot_id(row):
            parts = [str(row['Block_ID'])]
            for criterion in self.lot_criteria:
                parts.append(str(row[criterion]) if criterion in row else "NA")
            return "_".join(parts)

        df_with_blocks['Lot_ID'] = df_with_blocks.apply(build_lot_id, axis=1)

        # Auditoría con Lógica Penalty
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
        
        # Etiquetado para detalle
        def label_joint_type(row):
            lot_data = audit[audit['Lot_ID'] == row['Lot_ID']]
            n_rej = lot_data['RT1_Rejects'].values[0] if not lot_data.empty else 0
            if row['RT1rej'] and pd.notna(row['RT2Date1']): return "🛠️ REPAIR DONE (RT2)"
            if row['RT1rej']: return "❌ REJECTED"
            if pd.notna(row['RTDate1']): return "✅ STANDARD (OK)"
            if n_rej == 1: return "🚨 PENALTY"
            if n_rej > 1: return "🧨 FULL AUDIT"
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
                impact = 1.0 if debts.get(l_id, 0) > 0 else 0.0
                # INYECCIÓN IA: Sumamos la probabilidad como criterio de desempate
                impact += row.get('AI_Fail_Prob', 0.0)
                
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

# --- UI ---
st.set_page_config(page_title="RT Optimizer", layout="wide", page_icon="🏗️")
st.title(":material/engineering: RT Optimizer | AI Engine")

# NOMBRE DEL MODELO ACTUALIZADO
MODEL_FILE = 'modelo_WS_xgboost.joblib'

uploaded_file = st.file_uploader("Upload Daily SQL Extraction (CSV)", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
    df_raw.columns = df_raw.columns.str.strip()
    
    required_cols = ['Joint_ID', 'Subc', 'Welder1', 'Line', 'location', 'MaterialType', 'WPS.1.Description', 
                     'Dateofweld', 'RTDate1', 'RT_Perc', 'RT1rej', 'RT2Date1', 'Jointsize', 'Thickness']
    
    new_cols = {col: req for col in df_raw.columns for req in required_cols if col.lower() == req.lower()}
    df_raw.rename(columns=new_cols, inplace=True)
    for col in df_raw.select_dtypes(['object']).columns: df_raw[col] = df_raw[col].astype(str).str.strip()
    
    # Exclude non-metallic
    df_raw = df_raw[df_raw['MaterialType'].str.upper() != 'PLASTIC'].copy()

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    subs_list = ["ALL"] + sorted(df_raw['Subc'].unique().tolist())
    selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
    
    # Utils
    def get_available_scopes(df, sub):
        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        locs = df['location'].unique() if sub == "ALL" else df[df['Subc'] == sub]['location'].unique()
        available = []
        for scope, vals in location_map.items():
            if any(loc in locs for loc in vals): available.append(scope)
        return ["ALL"] + available if len(available) > 1 else available

    scopes = get_available_scopes(df_raw, selected_sub)
    location_scope = st.sidebar.radio("Location Scope:", options=scopes, index=0)
    days_per_lot = st.sidebar.number_input("Days per Window", min_value=1, value=14)
    fallback_perc = st.sidebar.slider("Fallback RT %", 0, 100, 10)

    st.sidebar.divider()
    db_criteria = [col for cond, col in zip([st.sidebar.checkbox("Subcontractor", value=True), st.sidebar.checkbox("Welder", value=True), st.sidebar.checkbox("Material Type", value=True), st.sidebar.checkbox("Welding Process", value=True), st.sidebar.checkbox("Line ID", value=False)], ['Subc', 'Welder1', 'MaterialType', 'WPS.1.Description', 'Line']) if cond]

    # --- MOTOR RUN ---
    engine = RTOptimizerEngine(fallback_perc/100, days_per_lot, db_criteria, location_scope, model_path=MODEL_FILE)
    
    df_to_process = df_raw.copy()
    if selected_sub != "ALL": df_to_process = df_raw[df_raw['Subc'] == selected_sub].copy()

    audit_df, df_with_lots = engine.get_lot_audit(df_to_process)

    if audit_df.empty:
        st.warning("No sampling data found.")
    else:
        tab1, tab2 = st.tabs([":material/assignment: Work Order (AI Optimized)", ":material/dashboard: Dashboard"])
        with tab1:
            if st.button("🚀 Generate Optimized Plan (Compliance + AI Risk)"):
                result = engine.execute_optimization(df_with_lots, audit_df.copy())
                if not result.empty:
                    result['Plan_Date'] = datetime.now().strftime('%d/%m/%Y')
                    st.write(f"Recommended Inspections: **{len(result)}**")
                    display_cols = ['Joint_ID', 'Risk_Level', 'Inspection_Reason', 'Lot_ID', 'Welder1', 'Line', 'MaterialType', 'WPS.1.Description', 'Jointsize', 'Thickness']
                    st.dataframe(result[[c for c in display_cols if c in result.columns]], use_container_width=True, hide_index=True)
                    st.download_button("📥 Download Plan", result.to_csv(sep=';', index=False).encode('utf-8-sig'), f"plan_v2_{selected_sub}.csv", "text/csv")
                else: st.success("✅ Compliance achieved.")

        with tab2:
            st.subheader(f"Dashboard: {selected_sub} | {location_scope}")
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
                det_cols = ['Joint_ID', 'Risk_Level', 'Inspection_Type', 'Line', 'Jointsize', 'Thickness', 'Dateofweld', 'RTDate1', 'RT2Date1', 'RT1rej']
                st.dataframe(df_with_lots[df_with_lots['Lot_ID'] == lot_id][[c for c in det_cols if c in df_with_lots.columns]], use_container_width=True, hide_index=True)
else:
    st.info("💡 Please upload your SQL CSV extraction.")