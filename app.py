import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIG ---
REQUIRED_COLS = [
    'Joint_ID','Subc','Welder1','Line','location','MaterialType',
    'WPS.1.Description','Dateofweld','RTDate1','RT_Perc',
    'RT1rej','RTAccepted','Jointsize','Thickness'
]

MODEL_FEATURES = [
    'Jointsize', 'Subc', 'MaterialType', 'WPS.1.Description', 'Thickness'
]

# --- LOAD MODEL ---
@st.cache_resource
def load_ai_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Model load failed: {e}")
    return None

# --- LOAD DATA ---
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file, sep=';', encoding='utf-8-sig')

    # Limpieza columnas (pero SIN renombrar)
    df.columns = df.columns.str.strip()

    # Crear columnas faltantes
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Limpieza de blancos → NaN
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})

    # Fechas
    df['Dateofweld'] = pd.to_datetime(df['Dateofweld'], dayfirst=True, errors='coerce')
    df['RTDate1'] = pd.to_datetime(df['RTDate1'], dayfirst=True, errors='coerce')

    df = df.dropna(subset=['Dateofweld']).copy()

    # Numéricos
    for col in ['Jointsize', 'Thickness', 'RT_Perc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Booleanos
    for col in ['RT1rej', 'RTAccepted']:
        df[col] = (
            df[col].astype(str).str.upper()
            .map({'TRUE': True, 'FALSE': False, '1': True, '0': False})
        ).fillna(False)

    return df


# --- ENGINE ---
class RTOptimizerEngine:

    def __init__(self, fallback_rt_perc, window_days, criteria, model):
        self.fallback_rt_perc = fallback_rt_perc
        self.window_days = window_days
        self.criteria = criteria
        self.model = model

    def run_full_process(self, df):

        d = df.copy()

        # Validar criterios
        valid_criteria = [c for c in self.criteria if c in d.columns]
        if not valid_criteria:
            st.error(f"No valid grouping columns. Available: {list(d.columns)}")
            st.stop()

        self.criteria = valid_criteria

        # RT %
        d['RT_Perc'] = d['RT_Perc'].replace(0, np.nan)
        d['RT_Perc'] = d['RT_Perc'].fillna(self.fallback_rt_perc * 100)

        # --- IA ---
        if self.model:
            try:
                X = d[MODEL_FEATURES].copy()

                # El modelo tolera NaN, pero aseguramos categóricos
                for col in ['Subc','MaterialType','WPS.1.Description']:
                    X[col] = X[col].fillna("UNKNOWN")

                d['AI_Prob'] = self.model.predict_proba(X)[:, 1]

            except Exception as e:
                st.warning(f"⚠️ Model failed: {e}")
                st.write("Columnas disponibles:", d.columns.tolist())
                d['AI_Prob'] = 0.0
        else:
            d['AI_Prob'] = 0.0

        # Riesgo
        d['Risk_Level'] = pd.cut(
            d['AI_Prob'],
            bins=[-1, 0.3, 0.75, 1],
            labels=["🟢 Low", "🟡 Medium", "🔴 High"]
        )

        # --- LOTES ---
        d = d.sort_values(self.criteria + ['Dateofweld'])

        def assign_blocks(group):
            group = group.copy()

            block = 0
            start = group['Dateofweld'].iloc[0]
            blocks = []

            for date in group['Dateofweld']:
                if date >= start + pd.Timedelta(days=self.window_days):
                    block += 1
                    start = date
                blocks.append(block)

            group['Block_ID'] = blocks

            group['Lot_ID'] = (
                group[self.criteria]
                .fillna("NA")
                .astype(str)
                .agg('_'.join, axis=1)
                + "_B"
                + group['Block_ID'].astype(str)
            )

            return group

        d = d.groupby(self.criteria, dropna=False).apply(assign_blocks).reset_index(drop=True)

        # --- PENALTY ---
        def label_group(group):

            group = group.sort_values(['RTDate1', 'Dateofweld'], na_position='last')

            fail_found = False
            tracer_count = 0
            force_100 = False

            types = []
            status = []

            for _, row in group.iterrows():

                if pd.isna(row['RTDate1']):
                    status.append("Not Inspected")
                elif not row['RT1rej']:
                    status.append("Standard RT")
                elif row['RTAccepted']:
                    status.append("Rejected & Repaired")
                else:
                    status.append("Rejected Pending")

                if force_100:
                    types.append("Penalty Lot 100%")
                    continue

                if row['RT1rej'] and not fail_found:
                    types.append("Random Inspection Joint")
                    fail_found = True

                elif fail_found and tracer_count < 2:
                    types.append("Penalty Tracer")
                    tracer_count += 1

                    if row['RT1rej']:
                        force_100 = True
                else:
                    types.append("Random Inspection Joint")

            group['Inspection_Type'] = types
            group['Inspection_Status'] = status

            return group

        d = d.groupby('Lot_ID', group_keys=False).apply(label_group)

        # --- AUDIT ---
        audit = d.groupby('Lot_ID').agg(
            Total_Joints=('Joint_ID', 'count'),
            RT1_Count=('RTDate1', 'count'),
            Rej_Count=('RT1rej', 'sum'),
            RT_Req=('RT_Perc', 'max')
        ).reset_index()

        def compute_required(lot_id):
            lot = d[d['Lot_ID'] == lot_id]

            if "Penalty Lot 100%" in lot['Inspection_Type'].values:
                return len(lot)

            base = np.ceil(len(lot) * (lot['RT_Perc'].iloc[0] / 100))
            return int(min(len(lot), base + 2 if lot['RT1rej'].any() else base))

        audit['Required'] = audit['Lot_ID'].apply(compute_required)
        audit['Deficit'] = (audit['Required'] - audit['RT1_Count']).clip(lower=0)

        return audit, d

    # --- OPTIMIZACIÓN ---
    def execute_optimization(self, df, audit):

        debts = audit.set_index('Lot_ID')['Deficit'].to_dict()

        candidates = df[df['RTDate1'].isna()].copy()
        candidates['Lot_Deficit'] = candidates['Lot_ID'].map(debts).fillna(0)

        candidates = candidates.sort_values(
            ['AI_Prob', 'Lot_Deficit'],
            ascending=[False, False]
        )

        plan = []

        for _, row in candidates.iterrows():
            lid = row['Lot_ID']

            if debts.get(lid, 0) > 0:
                debts[lid] -= 1
                row['Reason'] = row['Inspection_Type']
                plan.append(row)

        return pd.DataFrame(plan)


# --- UI ---
st.set_page_config(page_title="RT Optimizer PRO", layout="wide")
st.title("🏗️ RT Optimizer PRO (Original Columns Mode)")

model = load_ai_model("modelo_welding_lgb.joblib")

file = st.file_uploader("Upload CSV", type="csv")

if file:
    df = load_and_preprocess(file)

    st.write("📊 Columnas detectadas:", df.columns.tolist())

    criteria = st.multiselect(
        "Grouping Criteria",
        ['Subc', 'Welder1', 'MaterialType', 'WPS.1.Description', 'Line'],
        default=['Subc', 'Welder1']
    )

    engine = RTOptimizerEngine(0.1, 14, criteria, model)

    audit, df_lots = engine.run_full_process(df)

    if st.button("Generate Optimized Plan"):
        plan = engine.execute_optimization(df_lots, audit)

        st.metric("Selected Joints", len(plan))
        st.dataframe(plan, use_container_width=True)