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
        """Asigna Bloque_ID dinámicamente. Asegura que la columna siempre exista."""
        if df.empty:
            df['Block_ID'] = None
            return df
        
        df = df.sort_values('Dateofweld')
        groups = df.groupby(self.lot_criteria)
        processed_chunks = []

        for _, group in groups:
            group = group.copy()
            block_ids = []
            start_date = group['Dateofweld'].iloc[0]
            current_block = 0
            
            for date in group['Dateofweld']:
                if pd.isna(date) or pd.isna(start_date):
                    block_ids.append(-1)
                elif date < start_date + pd.Timedelta(days=self.window_days):
                    block_ids.append(current_block)
                else:
                    start_date = date
                    current_block += 1
                    block_ids.append(current_block)
            
            group['Block_ID'] = block_ids
            processed_chunks.append(group)
        
        if not processed_chunks:
            df['Block_ID'] = -1
            return df
            
        return pd.concat(processed_chunks)

    def get_lot_audit(self, df):
        d = df.copy()
        d['Dateofweld'] = pd.to_datetime(d['Dateofweld'], dayfirst=True, errors='coerce')
        d['RTDate1'] = pd.to_datetime(d['RTDate1'], dayfirst=True, errors='coerce')
        
        # Eliminar juntas sin fecha de soldadura (No se pueden procesar)
        d = d.dropna(subset=['Dateofweld'])
        if d.empty: return pd.DataFrame(), pd.DataFrame()
        
        d['RT_Perc'] = pd.to_numeric(d['RT_Perc'], errors='coerce')
        d['RT_Perc'] = d['RT_Perc'].replace(0, np.nan).fillna(self.fallback_rt_perc * 100)
        
        # Location Scope
        location_map = {"WS": ["YWS", "S", "WS"], "FW": ["YFW", "FW", "F"], "PL": ["PL"]}
        allowed_values = location_map.get(self.scope, [])
        
        df_filtered = d[(d['location'].isin(allowed_values)) & (d['RT_Perc'] < 100)].copy()
        if df_filtered.empty: return pd.DataFrame(), pd.DataFrame()

        # Dynamic Blocks
        df_with_blocks = self._assign_dynamic_blocks(df_filtered)
        
        # Seguridad: Si por algún motivo Block_ID no existe, lo creamos
        if 'Block_ID' not in df_with_blocks.columns:
            df_with_blocks['Block_ID'] = 0

        # Identidad del Lote
        def build_lot_id(row):
            parts = [str(row['Block_ID'])]
            for criterion in self.lot_criteria:
                val = str(row[criterion]) if criterion in row else "NA"
                parts.append(val)
            return "_".join(parts)

        df_with_blocks['Lot_ID'] = df_with_blocks.apply(build_lot_id, axis=1)

        # Auditoría
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
        return audit[cols_order], df_with_blocks

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
def get_available_scopes_for_sub(df, sub_name):
    ws_vals = ["YWS", "S", "WS"]
    fw_vals = ["YFW", "FW", "F"]
    pl_vals = ["PL"]
    sub_locs = df[df['Subc'] == sub_name]['location'].unique()
    available = []
    if any(loc in sub_locs for loc in ws_vals): available.append("WS")
    if any(loc in sub_locs for loc in fw_vals): available.append("FW")
    if any(loc in sub_locs for loc in pl_vals): available.append("PL")
    return available

# --- USER INTERFACE ---
st.set_page_config(page_title="RT Optimizer", layout="wide")
st.title(":material/engineering: RT Optimizer")

# Sidebar Configuration
st.sidebar.header(":material/settings: Configuration")
uploaded_file = st.file_uploader("Upload Daily SQL Extraction (CSV)", type="csv")

if uploaded_file:
    df_input = pd.read_csv(uploaded_file, sep=';')
    df_input.columns = df_input.columns.str.strip()
    
    # NORMALIZACIÓN DE CABECERAS
    required_cols = ['Joint_ID', 'Subc', 'Welder1', 'Line', 'location', 'MaterialType', 'WPS.1.Description', 'Dateofweld', 'RTDate1', 'RT_Perc']
    new_cols = {}
    for col in df_input.columns:
        for req in required_cols:
            if col.lower() == req.lower():
                new_cols[col] = req
    df_input.rename(columns=new_cols, inplace=True)
    
    missing = [c for c in required_cols if c not in df_input.columns]
    
    if missing:
        st.error(f"❌ Columns missing: {missing}")
    else:
        for col in df_input.select_dtypes(['object']).columns:
            df_input[col] = df_input[col].astype(str).str.strip()

        subs_list = sorted(df_input['Subc'].unique())
        selected_sub = st.sidebar.selectbox("🎯 Target Subcontractor:", options=subs_list)
        available_scopes = get_available_scopes_for_sub(df_input, selected_sub)
        
        if available_scopes:
            location_scope = st.sidebar.radio(f"Location Scope for {selected_sub}:", options=available_scopes)
            days_per_lot = st.sidebar.number_input("Days per Window", min_value=1, value=14)
            fallback_percentage = st.sidebar.slider("Fallback RT %", 0, 100, 10)
            
            st.sidebar.divider()
            c_subc = st.sidebar.checkbox("Subcontractor (Subc)", value=True)
            c_welder = st.sidebar.checkbox("Welder ID (Welder1)", value=True)
            c_material = st.sidebar.checkbox("Material (MaterialType)", value=True)
            c_process = st.sidebar.checkbox("Process (WPS.1.Description)", value=True)
            c_line = st.sidebar.checkbox("Line ID (Line)", value=False)
            db_criteria_map = [col for cond, col in zip([c_subc, c_welder, c_material, c_process, c_line], ['Subc', 'Welder1', 'MaterialType', 'WPS.1.Description', 'Line']) if cond]

            # ENGINE
            engine = RTOptimizerEngine(fallback_percentage/100, days_per_lot, db_criteria_map, location_scope)
            audit_df, df_with_lots = engine.get_lot_audit(df_input)

            if audit_df.empty:
                st.warning(f"⚠️ No juntas found for scope '{location_scope}'. Check 'Dateofweld' values.")
            else:
                tab1, tab2 = st.tabs([":material/assignment: Work Order", ":material/dashboard: Dashboard"])
                with tab1:
                    st.subheader(f"Inspection Plan: {selected_sub}")
                    sub_df_lots = df_with_lots[df_with_lots['Subc'] == selected_sub]
                    sub_audit = audit_df[audit_df['Subcontractor'] == selected_sub]
                    if st.button("🚀 Generate Plan"):
                        result = engine.execute_optimization(sub_df_lots, sub_audit.copy())
                        if not result.empty:
                            result['Plan_Date'] = datetime.now().strftime('%d/%m/%Y')
                            st.write(f"Recommended Inspections: **{len(result)}**")
                            display_cols = ['Joint_ID', 'Plan_Date', 'Lot_ID', 'Welder1', 'Line', 'MaterialType', 'WPS.1.Description', 'Dateofweld', 'RT_Perc']
                            st.dataframe(result[[c for c in display_cols if c in result.columns]], use_container_width=True, hide_index=True)
                            csv_data = result.to_csv(sep=';', index=False).encode('utf-8-sig')
                            st.download_button("📥 Download Plan", csv_data, f"plan_{selected_sub}.csv", "text/csv")
                        else:
                            st.success("✅ Compliance achieved.")

                with tab2:
                    dash_audit = audit_df.copy() if (view_option := st.selectbox("📊 Dashboard View:", options=["ALL"] + subs_list, index=subs_list.index(selected_sub)+1)) == "ALL" else audit_df[audit_df['Subcontractor'] == view_option]
                    k1, k2, k3, k4, k5 = st.columns(5)
                    total_l = len(dash_audit)
                    open_l = len(dash_audit[dash_audit['Status'] == '🔴 OPEN'])
                    k1.metric("Total Lots", total_l); k2.metric("Open Lots", open_l)
                    k3.metric("Compliance Rate", f"{((total_l-open_l)/total_l)*100:.1f}%" if total_l > 0 else "0%")
                    k4.metric("Avg. Actual RT %", f"{dash_audit['Current_RT_Done_%'].mean():.1f}%" if total_l > 0 else "0%")
                    k5.metric("Avg. Target RT %", f"{dash_audit['Current_RT_Req'].mean():.1f}%" if total_l > 0 else "0%")
                    st.divider()
                    selection_event = st.dataframe(dash_audit, use_container_width=True, on_select="rerun", selection_mode="single-row", hide_index=True)
                    if selection_event.selection.rows:
                        row_idx = selection_event.selection.rows[0]
                        lot_data = dash_audit.iloc[row_idx]
                        st.markdown(f"### 🔍 Detailed Explorer: Lot `{lot_data['Lot_ID']}`")
                        st.dataframe(df_with_lots[df_with_lots['Lot_ID'] == lot_data['Lot_ID']][['Joint_ID', 'Subc', 'Welder1', 'Line', 'Dateofweld', 'RTDate1', 'RT_Perc']], use_container_width=True, hide_index=True)

else:
    st.info("💡 **Awaiting Data.** Please upload your SQL CSV extraction.")
    st.markdown("### Required CSV Data Structure")
    schema_df = pd.DataFrame({
        'Mandatory Column': ['Joint_ID', 'Subc', 'Welder1', 'Line', 'location', 'MaterialType', 'WPS.1.Description', 'Dateofweld', 'RTDate1', 'RT_Perc'],
        'Description': ['ID', 'Subcontractor', 'Welder', 'Line No.', 'Scope', 'Material', 'Process', 'DD/MM/YYYY', 'Done Date', 'Target %']
    })
    st.table(schema_df)
    st.markdown("""
    **Processing Rules:**
    *   **Mandatory:** Rows with empty `Dateofweld` are ignored.
    *   **Max Rule:** If a lot has mixed requirements, the highest value is applied.
    *   **Fallback:** `RT_Perc` = 0 or Null is filled using the sidebar slider.
    """)