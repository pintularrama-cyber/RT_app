import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# --- CONFIG ---
DB_MAP = {
    'Joint_ID': 'Joint_ID', 'Subc': 'Subc', 'Welder1': 'Welder1', 'Line': 'Line',
    'location': 'location', 'MaterialType': 'MaterialType', 'WPS.1.Description': 'Process',
    'Dateofweld': 'Dateofweld', 'RTDate1': 'RTDate1', 'RT_Perc': 'RT_Perc',
    'RT1rej': 'RT1rej', 'RTAccepted': 'RTAccepted', 'Jointsize': 'Jointsize', 'Thickness': 'Thickness'
}

# --- LOADERS ---
@st.cache_resource
def load_ai_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Model load failed: {e}")
    return None

@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    # Rename dinámico
    rename_map = {col: DB_MAP[k] for col in df.columns for k in DB_MAP if col.lower() == k.lower()}
    df.rename(columns=rename_map, inplace=True)

    # Columnas obligatorias seguras
    for col in ['RTDate1', 'RT1rej', 'RTAccepted', 'RT_Perc']:
        if col not in df.columns:
            df[col] = np.nan

    if 'Joint_ID' not in df.columns:
        df['Joint_ID'] = range(1, len(df) + 1)

    # Limpieza
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    df['Dateofweld'] = pd.to_datetime(df['Dateofweld'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Dateofweld'])

    return df


# --- ENGINE ---
class RTOptimizerEngine:

    def __init__(self, fallback_rt_perc, window_days, criteria, scope, model):
        self.fallback_rt_perc = fallback_rt_perc
        self.window_days = window_days
        self.criteria = criteria
        self.scope = scope
        self.model = model

    def run_full_process(self, df):

        if not self.criteria:
            st.error("Select at least one grouping criteria")
            st.stop()

        d = df.copy()

        # --- SAFE CONVERSIONS ---
        d['RTDate1'] = pd.to_datetime(d['RTDate1'], errors='coerce')

        for col in ['Jointsize', 'Thickness']:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)

        d['RT_Perc'] = pd.to_numeric(d['RT_Perc'], errors='coerce')
        d['RT_Perc'] = d['RT_Perc'].replace(0, np.nan).fillna(self.fallback_rt_perc * 100)

        for col in ['RT1rej', 'RTAccepted']:
            d[col] = d[col].astype(str).str.upper().map({'TRUE': True, 'FALSE': False, '1': True, '0': False}).fillna(False)

        # --- FILTRO ---
        d = d[d['RT_Perc'] < 100].copy()
        if d.empty:
            return pd.DataFrame(), pd.DataFrame()

        # --- IA ---
        if self.model:
            try:
                X = d[['Jointsize', 'Subc', 'MaterialType', 'Process', 'Thickness']]
                d['AI_Prob'] = self.model.predict_proba(X)[:, 1]
            except Exception as e:
                st.warning(f"Model prediction failed: {e}")
                d['AI_Prob'] = 0.0
        else:
            d['AI_Prob'] = 0.0

        # --- RISK LABEL ---
        d['Risk_Level'] = pd.cut(
            d['AI_Prob'],
            bins=[-1, 0.3, 0.75, 1],
            labels=["🟢 Low", "🟡 Medium", "🔴 High"]
        )

        # --- LOT GENERATION ---
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
            group['Lot_ID'] = group[self.criteria].astype(str).agg('_'.join, axis=1) + "_B" + group['Block_ID'].astype(str)

            return group

        d = d.groupby(self.criteria, dropna=False).apply(assign_blocks).reset_index(drop=True)

        # --- PENALTY LOGIC CONSISTENTE ---
        def label_group(group):

            group = group.sort_values(['RTDate1', 'Dateofweld'], na_position='last')

            fail_found = False
            tracer_count = 0
            force_100 = False

            types = []
            status = []

            for _, row in group.iterrows():

                # STATUS
                if pd.isna(row['RTDate1']):
                    status.append("Not Inspected")
                elif not row['RT1rej']:
                    status.append("Standard RT")
                elif row['RTAccepted']:
                    status.append("Rejected & Repaired")
                else:
                    status.append("Rejected Pending")

                # TYPE
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

        # Required basado en lógica real
        def compute_required(lot_id):
            lot = d[d['Lot_ID'] == lot_id]

            if "Penalty Lot 100%" in lot['Inspection_Type'].values:
                return len(lot)

            base = np.ceil(len(lot) * (lot['RT_Perc'].iloc[0] / 100))
            return int(min(len(lot), base + 2 if lot['RT1rej'].any() else base))

        audit['Required'] = audit['Lot_ID'].apply(compute_required)
        audit['Deficit'] = (audit['Required'] - audit['RT1_Count']).clip(lower=0)

        return audit, d

    # 🔥 PRIORIDAD: IA → DEFICIT
    def execute_optimization(self, df, audit):

        debts = audit.set_index('Lot_ID')['Deficit'].to_dict()

        candidates = df[df['RTDate1'].isna()].copy()
        candidates['Lot_Deficit'] = candidates['Lot_ID'].map(debts).fillna(0)

        # 🔥 TU REGLA
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
st.title("🏗️ RT Optimizer PRO")

model = load_ai_model("modelo_welding_lgb.joblib")

file = st.file_uploader("Upload CSV", type="csv")

if file:
    df = load_and_preprocess(file)

    criteria = st.multiselect(
        "Grouping Criteria",
        ['Subc', 'Welder1', 'MaterialType', 'Process', 'Line'],
        default=['Subc', 'Welder1']
    )

    engine = RTOptimizerEngine(0.1, 14, criteria, "ALL", model)

    audit, df_lots = engine.run_full_process(df)

    if st.button("Generate Plan"):
        plan = engine.execute_optimization(df_lots, audit)

        st.metric("Selected Joints", len(plan))
        st.dataframe(plan)