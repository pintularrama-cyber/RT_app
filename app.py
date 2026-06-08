import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sklearn.compose
import sklearn.impute
from datetime import datetime
import time

# --- 1. PARCHES DE COMPATIBILIDAD IA ---
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

if not hasattr(sklearn.impute.SimpleImputer, '_fill_dtype'):
    setattr(sklearn.impute.SimpleImputer, '_fill_dtype', float)

# --- 2. CONFIGURACIÓN DE COLUMNAS (NOMBRES ORIGINALES CSV) ---
REQUIRED_COLS = [
    'Joint_ID', 'Subc', 'Welder1', 'Line', 'location', 
    'MaterialType', 'WPS.1.Description', 'Dateofweld', 
    'RTDate1', 'RT_Perc', 'RT1rej', 'RTAccepted', 
    'Jointsize', 'Thickness'
]

# --- 3. FUNCIONES BASE ---
@st.cache_resource
def load_ai_model(path):
    if os.path.exists(path):
        try: return joblib.load(path)
        except: return None
    return None

@st.cache_data
def load_and_preprocess_data(file):
    # utf-8-sig para limpiar el BOM del Joint_ID
    d = pd.read_csv(file, sep=';', encoding='utf-8-sig')
    d.columns = d.columns.str.strip()
    
    # Comprobar que no falten columnas críticas
    missing = [c for c in REQUIRED_COLS if c not in d.columns]
    if missing:
        st.error(f"❌ Columns missing in CSV: {missing}")
        st.stop()

    for col in d.select_dtypes(['object']).columns:
        d[col] = d[col].astype(str).str.strip()
    
    d['Dateofweld'] = pd.to_datetime(d['Dateofweld'], dayfirst=True, errors='coerce')
    d = d.dropna(subset=['Dateofweld']).copy()
    d = d[d['MaterialType'].str.upper() != 'PLASTIC'].copy()
    return d

# --- 4. OPTIMIZATION ENGINE ---
class RTOptimizerEngine:
    def __init__(self, fallback_rt_perc, window_days, lot_criteria, selected_location_scope, model):
        self.fallback_rt_perc = fallback_rt_perc
        self.window_days = window_days
        self.lot_criteria = lot_criteria 
        self.scope = selected_location_scope
        self.model_pipeline = model

    def get_lot_audit(self, df):
        d = df.copy()
        d['RTDate1'] = pd.to_datetime(d['RTDate1'], dayfirst=True, errors='coerce')
        
        # Saneamiento de comas para el Pipeline de ML
        for col in ['Jointsize', 'Thickness']:
            d[col] = d[col].astype(str).str.replace(',', '.')
            d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0)
            
        d['RT_Perc'] = pd.to_numeric(d['RT_Perc'], errors='coerce').replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        
        for col in ['RT1rej', 'RTAccepted']:
            d[col] = d[col].astype(str).str.upper().str.strip().map({'TRUE': True, 'FALSE': False, '1': True, '0': False, 'NAN': False}).fillna(False)

        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        if self.scope == "ALL": df_filtered = d[d['RT_Perc'] < 100].copy()
        else:
            allowed = location_map.get(self.scope, [])
            df_filtered = d[(d['location'].isin(allowed)) & (d['RT_Perc'] < 100)].copy()
        
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame()

        # IA Inferencia
        if self.model_pipeline:
            try:
                X_feats = ['Jointsize', 'Subc', 'MaterialType', 'WPS.1.Description', 'Thickness']
                df_filtered['AI_Prob'] = self.model_pipeline.predict_proba(df_filtered[X_feats])[:, 1]
            except Exception as e:
                st.error(f"ML Error: {e}")
                df_filtered['AI_Prob'] = 0.0
        else: df_filtered['AI_Prob'] = 0.0

        df_filtered['Risk_Level'] = df_filtered['AI_Prob'].apply(lambda p: f"🔴 High ({p*100:.1f}%)" if p > 0.75 else (f"🟡 Med ({p*100:.1f}%)" if p > 0.3 else f"🟢 Low ({p*100:.1f}%)"))

        # Lotes Dinámicos
        df_filtered = df_filtered.sort_values(['Subc', 'Dateofweld'])
        processed = []
        for _, group in df_filtered.groupby(self.lot_criteria, dropna=False):
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
            prefix = "_".join([str(group[c].iloc[0]) for c in self.lot_criteria])
            group['Lot_ID'] = group['Block_ID'].astype(str) + "_" + prefix
            processed.append(group)
        df_wb = pd.concat(processed).reset_index(drop=True)

        # Identificar juntas rechazadas y no aceptadas (pendientes de reparación)
        df_wb['Is_Pending_Repair'] = (df_wb['RT1rej'] == True) & (df_wb['RTAccepted'] == False)

        # Auditoría Base
        audit = df_wb.groupby('Lot_ID', as_index=False).agg(
            Total_Joints=('Joint_ID', 'count'), 
            RT1_Count=('RTDate1', 'count'), 
            Rej_Count=('RT1rej', 'sum'), 
            Pending_Repairs=('Is_Pending_Repair', 'sum'),
            RT_Req=('RT_Perc', 'max'), 
            Welder1=('Welder1', 'first'), 
            Subc=('Subc', 'first'), 
            MaterialType=('MaterialType', 'first'), 
            Block_Start=('Dateofweld', 'min')
        )

        # Etiquetado cronológico (Penalizaciones)
        df_wb = df_wb.merge(audit[['Lot_ID', 'Rej_Count']], on='Lot_ID', how='left')
        
        final_types = []; final_status = []
        current_lot = ""; found_fail = False; tracers = 0; is_100 = False
        df_wb = df_wb.sort_values(['Lot_ID', 'RTDate1', 'Dateofweld'], ascending=[True, True, True])

        for _, row in df_wb.iterrows():
            if row['Lot_ID'] != current_lot:
                current_lot = row['Lot_ID']; found_fail = False; tracers = 0; is_100 = row['Rej_Count'] > 1
            
            if pd.isna(row['RTDate1']): s = "Not Inspected"
            elif not row['RT1rej']: s = "RT Accepted"
            elif row['RTAccepted']: s = "Rejected & Repaired"
            else: s = "Rejected & Pending to be repaired"
            final_status.append(s)

            if is_100: final_types.append("Penalty Lot 100%")
            elif row['RT1rej'] and not found_fail:
                final_types.append("Random Inspection Joint"); found_fail = True
            elif found_fail and tracers < 2:
                final_types.append("Penalty Tracer"); tracers += 1
                if row['RT1rej']: is_100 = True
            else: final_types.append("Random Inspection Joint")

        df_wb['Inspection_Type'] = final_types
        df_wb['Inspection_Status'] = final_status

        # === NUEVO: CÁLCULO DE LAS COLUMNAS DE PENALIZACIÓN DEL LOTE ===
        # Agrupamos df_wb para comprobar si se asignó "Penalty Tracer" o "Penalty Lot 100%" a alguna junta del lote
        penalties_summary = df_wb.groupby('Lot_ID').agg(
            has_tracer=('Inspection_Type', lambda x: 'Penalty Tracer' in x.values),
            has_100=('Inspection_Type', lambda x: 'Penalty Lot 100%' in x.values)
        ).reset_index()

        penalties_summary['Penalty Tracer'] = np.where(penalties_summary['has_tracer'], 'Yes', 'No')
        penalties_summary['100% Penalty'] = np.where(penalties_summary['has_100'], 'Yes', 'No')
        # ==============================================================

        def final_req(r):
            lot_data = df_wb[df_wb['Lot_ID'] == r['Lot_ID']]
            if "Penalty Lot 100%" in lot_data['Inspection_Type'].values: return r['Total_Joints']
            base = np.ceil(r['Total_Joints'] * (r['RT_Req'] / 100))
            if r['Rej_Count'] >= 1: return int(min(r['Total_Joints'], base + 2))
            return int(base)

        audit['Required'] = audit.apply(final_req, axis=1)
        audit['Deficit'] = (audit['Required'] - audit['RT1_Count']).clip(lower=0)
        audit['Done_%'] = (audit['RT1_Count'] / audit['Total_Joints'] * 100).round(1)
        audit['Status'] = np.where((audit['Deficit'] == 0) & (audit['Pending_Repairs'] == 0), '🟢 CLOSED', '🔴 OPEN')
        
        # NUEVO: Combinamos las nuevas columnas con la tabla principal de auditoría (audit)
        audit = audit.merge(penalties_summary[['Lot_ID', 'Penalty Tracer', '100% Penalty']], on='Lot_ID', how='left')

        return audit, df_wb

    def execute_optimization(self, df_wb, audit):
        if audit.empty: return pd.DataFrame()
        debts = audit.set_index('Lot_ID')['Deficit'].to_dict()
        candidates = df_wb[df_wb['RTDate1'].isna()].sort_values('AI_Prob', ascending=False)
        plan = []
        for _, row in candidates.iterrows():
            l_id = row['Lot_ID']
            if debts.get(l_id, 0) > 0:
                debts[l_id] -= 1
                row['Reason'] = row['Inspection_Type']
                plan.append(row)
        return pd.DataFrame(plan)

# --- UI ---
st.set_page_config(page_title="RT Optimizer", layout="wide", page_icon="🏗️")
st.title(":material/engineering: RT Optimizer")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_welding_lgb.joblib')
loaded_model = load_ai_model(MODEL_PATH)


# Título general para las dos opciones de entrada de datos
st.write("**Select Data Source / Extraction Method:**")

# Creamos 3 columnas: una para el cargador, otra estrecha para el "OR", y otra para la base de datos
col_upload, col_or, col_db = st.columns([10, 2, 8], vertical_alignment="center")

with col_db:
    # Botón de conexión a la base de datos (a la izquierda)
    connect_clicked = st.button("🔌 Connect with PCA Database", use_container_width=True)
    if connect_clicked:
        st.toast("⚡ Connection feature coming soon! Please use CSV upload for now.", icon="🔌")

with col_or:
    # Texto "OR" estilizado y centrado en el medio
    st.markdown("<div style='text-align: center; font-weight: bold; color: gray; font-size: 1.1rem;'>OR</div>", unsafe_allow_html=True)

with col_upload:
    # Cargador de archivos CSV (a la derecha) con etiqueta oculta para mantener la alineación vertical
    uploaded_file = st.file_uploader(
        "Upload Daily SQL Extraction", 
        type="csv", 
        label_visibility="collapsed"
    )


if uploaded_file:
    # 1. Simulación de delay de 3 segundos (solo la primera vez que se lee este archivo)
    if loaded_model:
        import time
        file_key = f"loaded_{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state:
            with st.sidebar:
                with st.spinner("Initializing AI Engine..."):
                    time.sleep(10)
            st.session_state[file_key] = True

    # 2. Renderizado del estado y del selector de modelos (SOLO si ya se subió el CSV)
    if loaded_model: 
        st.sidebar.success("✅ ML Engine Active")
        selected_model = st.sidebar.selectbox("🤖 Selected AI Model:", options=["General", "Workshop"])
        if selected_model == "Workshop":
            st.sidebar.info("💡 Prototype: Running 'General' engine under the hood.")
    else: 
        st.sidebar.warning("⚠️ Standard Mode Active")

if uploaded_file:
    df_raw = load_and_preprocess_data(uploaded_file)
    subs_list = ["ALL"] + sorted(df_raw['Subc'].unique().tolist())
    selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
    
    def get_scopes(df, sub):
        m = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        locs = df['location'].unique() if sub == "ALL" else df[df['Subc'] == sub]['location'].unique()
        available = [s for s, v in m.items() if any(l in locs for l in v)]
        return ["ALL"] + available if len(available) > 1 else available

    location_scope = st.sidebar.radio("Location Scope:", options=get_scopes(df_raw, selected_sub), index=0)

    # 1. HARDCODE: Definimos el Fallback RT % directamente al 10% (0.10) sin el slider anterior
    fallback_perc_fixed = 0.10

    # Línea divisoria
    st.sidebar.divider()

    # 3. TEXTO EXPLICATIVO: Mensaje introductorio para los criterios de los lotes
    st.sidebar.markdown("### Lot Grouping Options")
    st.sidebar.write("Please select joint lot creation criteria:")

    # 2. REUBICACIÓN: El selector "Days per Window" ahora se muestra debajo de la línea
    days_per_lot = st.sidebar.number_input("Days per Window", min_value=1, value=14)

    # Checkboxes de selección de criterios
    db_criteria = [
        col for cond, col in zip([
            st.sidebar.checkbox("Subcontractor (Subc)", value=True), 
            st.sidebar.checkbox("Welder ID (Welder1)", value=True), 
            st.sidebar.checkbox("Material Type", value=True), 
            st.sidebar.checkbox("Welding Process", value=True), 
            st.sidebar.checkbox("Line ID", value=False)
        ], ['Subc', 'Welder1', 'MaterialType', 'WPS.1.Description', 'Line']) if cond
    ]
    engine = RTOptimizerEngine(fallback_perc_fixed, days_per_lot, db_criteria, location_scope, loaded_model)
    df_input = df_raw.copy() if selected_sub == "ALL" else df_raw[df_raw['Subc'] == selected_sub].copy()
    audit_df, df_with_lots = engine.get_lot_audit(df_input)

    if audit_df.empty: st.warning("No sampling data found.")
    else:
        tab1, tab2 = st.tabs([":material/assignment: Work Order", ":material/dashboard: Dashboard"])
        with tab1:
            if st.button("🚀 Generate Optimized Plan"):
                result = engine.execute_optimization(df_with_lots, audit_df.copy())
                if not result.empty:
                    result['Plan_Date'] = datetime.now().strftime('%d/%m/%Y')
                    st.dataframe(result[['Joint_ID', 'Risk_Level', 'Reason', 'Lot_ID', 'Welder1', 'Line', 'MaterialType', 'WPS.1.Description']], use_container_width=True, hide_index=True)
                    st.download_button("📥 Download Plan", result.to_csv(sep=';', index=False).encode('utf-8-sig'), f"plan_{selected_sub}.csv")
                else: st.success("✅ Compliance achieved.")

        with tab2:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Lots", len(audit_df)); k2.metric("Open Lots", len(audit_df[audit_df['Status'] == '🔴 OPEN']))
            k3.metric("Project Compliance", f"{(len(audit_df[audit_df['Status'] == '🟢 CLOSED'])/len(audit_df)*100 if len(audit_df)>0 else 0):.1f}%")
            k4.metric("Avg. Actual RT %", f"{audit_df['Done_%'].mean():.1f}%"); k5.metric("Avg. Target RT %", f"{audit_df['RT_Req'].mean():.1f}%")
            st.divider()
            f1, f2, f3, f4 = st.columns(4)
            with f1: sl = st.multiselect("Filter Lot ID", options=sorted(audit_df['Lot_ID'].unique()))
            with f2: sw = st.multiselect("Filter Welder", options=sorted(audit_df['Welder1'].unique()))
            with f3: sm = st.multiselect("Filter Material", options=sorted(audit_df['MaterialType'].unique()))
            with f4: ss = st.multiselect("Filter Status", options=['🔴 OPEN', '🟢 CLOSED'])
            f_a = audit_df.copy()
            if sl: f_a = f_a[f_a['Lot_ID'].isin(sl)]
            if sw: f_a = f_a[f_a['Welder1'].isin(sw)]
            if sm: f_a = f_a[f_a['MaterialType'].isin(sm)]
            if ss: f_a = f_a[f_a['Status'].isin(ss)]

            # === NUEVO: Subtítulo y Botón de Descarga alineados en paralelo ===
            col_header, col_download = st.columns([3, 1], vertical_alignment="bottom")
            with col_header:
                st.subheader("All Lots Summary Log")
            with col_download:
                st.download_button(
                    label="📥 Download Lots Summary Log",
                    data=f_a.to_csv(sep=';', index=False).encode('utf-8-sig'),
                    file_name=f"lots_summary_{selected_sub}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            # =================================================================

            event = st.dataframe(f_a, use_container_width=True, on_select="rerun", selection_mode="single-row", hide_index=True)
            if event.selection.rows:
                row_idx = event.selection.rows[0]; lid = f_a.iloc[row_idx]['Lot_ID']
                
                # === NUEVO: Título del Detalle y Botón de Descarga en paralelo (ARRIBA) ===
                col_det_header, col_det_download = st.columns([3, 1], vertical_alignment="bottom")
                with col_det_header:
                    st.markdown(f"### 🔍 Detailed Explorer: Lot `{lid}`")
                with col_det_download:
                    st.download_button(
                        label="📥 Download Detail",
                        data=df_with_lots[df_with_lots['Lot_ID'] == lid].to_csv(sep=';', index=False).encode('utf-8-sig'),
                        file_name=f"detail_{lid}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                # ========================================================================
                
                det_cols = ['Joint_ID', 'Risk_Level', 'Inspection_Type', 'Inspection_Status', 'Line', 'Dateofweld', 'RTDate1', 'RT1rej', 'RTAccepted']
                
                # Filtramos los datos del lote seleccionado
                df_detail = df_with_lots[df_with_lots['Lot_ID'] == lid][det_cols].copy()
                
                # Definimos la función para pintar las filas basadas en la columna 'Inspection_Type'
                def highlight_penalty_rows(row):
                    style = ''
                    if row['Inspection_Type'] == 'Penalty Tracer':
                        style = 'background-color: #FFE082; color: black;'
                    elif row['Inspection_Type'] == 'Penalty Lot 100%':
                        style = 'background-color: #FFCDD2; color: black;'
                    return [style] * len(row)

                # Aplicamos el estilo al DataFrame
                styled_detail = df_detail.style.apply(highlight_penalty_rows, axis=1)
                
                # Renderizamos el objeto estilizado con st.dataframe (YA SIN el botón al final)
                st.dataframe(styled_detail, use_container_width=True, hide_index=True)