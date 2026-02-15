import streamlit as st
import pandas as pd
import numpy as np
import plspm.config as c
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import time
import logging
import os

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger(__name__)

st.set_page_config(page_title="Dashboard Analisis PLS-SEM", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 24px; font-weight: bold; color: #2c3e50; margin-bottom: 20px;}
    .sub-header {font-size: 18px; font-weight: bold; color: #34495e; margin-top: 20px; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px;}
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #2980b9; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("Dashboard Analisis PLS-SEM (Final)")

def get_optimal_processes(bootstrap_iterations):
    n_cores = os.cpu_count()
    for i in range(n_cores, 0, -1):
        if bootstrap_iterations % i == 0:
            return i
    return 1

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('last clean data.csv', delimiter=';')
        data.columns = data.columns.str.strip()
        return data
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None

def calculate_htmt(data, mv_map):
    lvs = sorted(list(set(mv_map.values())))
    htmt_matrix = pd.DataFrame(index=lvs, columns=lvs)
    model_indicators = list(mv_map.keys())
    corr_matrix = data[model_indicators].corr().abs()

    for i in lvs:
        for j in lvs:
            if i == j:
                htmt_matrix.loc[i, j] = 1.0
                continue
            indicators_i = [k for k, v in mv_map.items() if v == i]
            indicators_j = [k for k, v in mv_map.items() if v == j]
            r_ij = corr_matrix.loc[indicators_i, indicators_j].values.mean()
            if len(indicators_i) > 1:
                vals_i = corr_matrix.loc[indicators_i, indicators_i].values
                r_ii = vals_i[~np.eye(vals_i.shape[0], dtype=bool)].mean()
            else:
                r_ii = 1.0
            if len(indicators_j) > 1:
                vals_j = corr_matrix.loc[indicators_j, indicators_j].values
                r_jj = vals_j[~np.eye(vals_j.shape[0], dtype=bool)].mean()
            else:
                r_jj = 1.0
            htmt_val = r_ij / np.sqrt(r_ii * r_jj)
            htmt_matrix.loc[i, j] = htmt_val
    htmt_matrix = htmt_matrix.apply(pd.to_numeric)
    mask = np.triu(np.ones(htmt_matrix.shape), k=0).astype(bool)
    htmt_matrix = htmt_matrix.where(~mask, np.nan)
    return htmt_matrix

def calculate_indicator_q2(data, scores, structure_paths, mv_map):
    q2_results = []
    endogenous_lvs = set([dst for _, dst in structure_paths])
    
    for lv in endogenous_lvs:
        indicators = [k for k, v in mv_map.items() if v == lv]
        predictors = [src for src, dst in structure_paths if dst == lv]
        if not predictors: continue
        X = scores[predictors].values
        
        for indicator in indicators:
            y = data[indicator].values
            kf = KFold(n_splits=7, shuffle=True, random_state=42)
            sse, sso = 0, 0
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_test)
                sse += np.sum((y_test - y_pred)**2)
                sso += np.sum((y_test - np.mean(y_train))**2)
                
            q2_val = 1 - (sse / sso) if sso > 0 else 0
            q2_results.append({'Variabel': indicator, 'Q²': q2_val})
            
    return pd.DataFrame(q2_results)

def calculate_indicator_vif(data, mv_map):
    model_indicators = list(mv_map.keys())
    vif_data = pd.DataFrame()
    vif_data["Indikator"] = model_indicators
    vif_values = [variance_inflation_factor(data[model_indicators].values, i) for i in range(len(model_indicators))]
    vif_data["VIF"] = vif_values
    return vif_data

def calculate_fornell_larcker(scores, ave_values):
    lv_correlations = scores.corr()
    sqrt_ave = np.sqrt(ave_values)
    
    lvs = lv_correlations.columns.tolist()
    
    fornell_larcker_matrix = pd.DataFrame(np.nan, index=lvs, columns=lvs)
    
    for i, row_lv in enumerate(lvs):
        for j, col_lv in enumerate(lvs):
            if j == i:
                fornell_larcker_matrix.loc[row_lv, col_lv] = sqrt_ave.get(row_lv)
            elif j < i:
                fornell_larcker_matrix.loc[row_lv, col_lv] = lv_correlations.loc[row_lv, col_lv]
    
    return fornell_larcker_matrix

def build_config(data, active_paths):
    structure = c.Structure()
    involved_lvs = set()
    for src, dst in active_paths:
        structure.add_path([src], [dst])
        involved_lvs.add(src); involved_lvs.add(dst)
    
    conf = c.Config(structure.path(), scaled=True)
    mv_map = {} 
    
    def add_lv(lv, prefix, mvs=None):
        if lv not in involved_lvs: return
        
        if mvs:
            cols = [m for m in mvs if m in data.columns]
        else:
            cols = [m for m in data.columns if m.startswith(prefix)]
        if cols:
            conf.add_lv(lv, Mode.A, *[c.MV(m) for m in cols])
            for m in cols: mv_map[m] = lv

    for lv in ["PE", "EE", "SI", "FC", "HM", "PV", "PR", "BI", "UB"]:
        add_lv(lv, lv)
    add_lv("H", "H", mvs=["H1", "H2", "H3", "H4"])
    
    return conf, mv_map

@st.cache_data
def run_analysis(df, path_definitions):
    main_config, mv_map = build_config(df, path_definitions)
    
    missing_cols = [col for col in mv_map.keys() if col not in df.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan di data: {', '.join(missing_cols)}")
        st.stop()

    bootstrap_iterations = 5000
    processes = get_optimal_processes(bootstrap_iterations)
    
    plspm_calc = Plspm(df, main_config, Scheme.PATH, iterations=300, tolerance=1e-7, bootstrap=True, bootstrap_iterations=bootstrap_iterations, processes=processes)
    
    boot_results = plspm_calc.bootstrap()
    scores = plspm_calc.scores()
    inner_sum = plspm_calc.inner_summary()
    unidim = plspm_calc.unidimensionality()

    loadings = plspm_calc.outer_model()
    loadings['Variabel'] = loadings.index.map(mv_map)

    htmt_matrix = calculate_htmt(df, mv_map)
    
    boot_paths = boot_results.paths().reset_index().rename(columns={'index': 'Jalur'})
    
    std_cols = [col for col in boot_paths.columns if 'std' in col.lower() or 'error' in col.lower()]
    if std_cols:
        boot_paths.rename(columns={std_cols[0]: 'Standard Deviation'}, inplace=True)
    else:
        boot_paths['Standard Deviation'] = 0.0
    
    orig_col = 'original' if 'original' in boot_paths.columns else boot_paths.columns[1]
    boot_paths.rename(columns={orig_col: 'Original Sample (O)'}, inplace=True)
    
    if 't stat.' not in boot_paths.columns:
         boot_paths['t stat.'] = (boot_paths['Original Sample (O)'] / boot_paths['Standard Deviation']).fillna(0)
    
    boot_paths['P-Values'] = 2 * (1 - t.cdf(boot_paths['t stat.'].abs(), df=499))
    boot_paths['Keputusan'] = np.where((boot_paths['P-Values'] < 0.05) & (boot_paths['t stat.'].abs() > 1.96), "DITERIMA", "DITOLAK")

    lvs_to_flip = []
    all_lvs = sorted(list(set(mv_map.values())))
    for lv in all_lvs:
        lv_loads = loadings[loadings['Variabel'] == lv]['loading']
        if lv_loads.mean() < 0:
            lvs_to_flip.append(lv)
    
    if lvs_to_flip:
        for lv in lvs_to_flip:
            loadings.loc[loadings['Variabel'] == lv, 'loading'] *= -1
            scores[lv] *= -1

        for idx, row in boot_paths.iterrows():
            path_str = row['Jalur']
            if "->" in path_str:
                src, dst = path_str.split(" -> ")
                multiplier = 1
                if src in lvs_to_flip: multiplier *= -1
                if dst in lvs_to_flip: multiplier *= -1
                
                if multiplier != 1:
                    boot_paths.at[idx, 'Original Sample (O)'] *= multiplier
                    boot_paths.at[idx, 't stat.'] *= multiplier

    fornell_larcker_matrix = calculate_fornell_larcker(scores, inner_sum['ave'])

    f2_values = []
    orig_r2 = inner_sum['r_squared']
    for i, (src, dst) in enumerate(path_definitions):
        temp_paths = [p for p in path_definitions if p != (src, dst)]
        temp_config, temp_mv_map = build_config(df, temp_paths)
        if not temp_mv_map: continue
        temp_calc = Plspm(df, temp_config, Scheme.PATH, iterations=100, bootstrap=False)
        r2_incl = orig_r2.get(dst, 0)
        r2_excl = temp_calc.inner_summary()['r_squared'].get(dst, 0)
        f2 = (r2_incl - r2_excl) / (1 - r2_incl) if (1 - r2_incl) != 0 else 0
        f2_values.append({'Jalur': f"{src} -> {dst}", 'f2': abs(f2)})
        
    final_results = pd.merge(boot_paths, pd.DataFrame(f2_values), on='Jalur', how='left')
    if 't stat.' in final_results.columns:
        final_results['t stat.'] = final_results['t stat.'].abs()
    final_results.to_csv('path_results.csv')

    indicator_vif_df = calculate_indicator_vif(df, mv_map)
    indicator_q2_df = calculate_indicator_q2(df, scores, path_definitions, mv_map)
    
    return {
        "final_results": final_results,
        "unidim": unidim,
        "inner_sum": inner_sum,
        "loadings": loadings,
        "mv_map": mv_map,
        "htmt_matrix": htmt_matrix,
        "fornell_larcker_matrix": fornell_larcker_matrix,
        "indicator_vif_df": indicator_vif_df,
        "indicator_q2_df": indicator_q2_df,
        "scores": scores,
    }

def run_analysis_with_progress(df, path_definitions):
    progress_bar = st.progress(0)
    st.info("Memulai analisis PLS-SEM...")
    logger.info("Memulai analisis PLS-SEM...")

    results = run_analysis(df, path_definitions)

    st.info("Menjalankan bootstrap...")
    logger.info("Running bootstrap...")
    progress_bar.progress(25)
    
    st.info("Menghitung HTMT...")
    logger.info("Calculating HTMT...")
    progress_bar.progress(50)

    st.info("Menghitung f-squared...")
    logger.info("Calculating f-squared...")
    progress_bar.progress(75)

    st.success("Analisis Selesai!")
    logger.info("Analisis Selesai!")
    progress_bar.progress(100)

    return results

def main():
    df = load_data()
    if df is None:
        st.stop()

    path_definitions = [
        ("PE", "BI"), ("EE", "BI"), ("SI", "BI"), ("FC", "BI"), 
        ("HM", "BI"), ("PV", "BI"), ("H",  "BI"), ("PR", "BI"),
        ("FC", "UB"), ("H",  "UB"), ("BI", "UB")
    ]
    
    if st.button("Rerun Perhitungan"):
        st.cache_data.clear()
        st.success("Cache telah dibersihkan, perhitungan akan diulang.")

    try:
        results = run_analysis_with_progress(df, path_definitions)
        
        final_results = results["final_results"]
        unidim = results["unidim"]
        inner_sum = results["inner_sum"]
        loadings = results["loadings"]
        mv_map = results["mv_map"]
        htmt_matrix = results["htmt_matrix"]
        fornell_larcker_matrix = results["fornell_larcker_matrix"]
        indicator_vif_df = results["indicator_vif_df"]
        indicator_q2_df = results["indicator_q2_df"]
        scores = results["scores"]
        
        tab1, tab2, tab3 = st.tabs(["1. Model Pengukuran (Outer Model)", "2. Model Struktural (Inner Model)", "3. Pengujian Hipotesis"])

        with tab1:
            st.markdown("### Validitas Konvergen & Reliabilitas")
            rel = pd.concat([unidim[['cronbach_alpha', 'dillon_goldstein_rho']], inner_sum['ave']], axis=1)
            rel.columns = ["Cronbach's Alpha", "Composite Reliability", "AVE"]
            st.dataframe(rel.round(3).style.format("{:.3f}"))

            st.markdown("### Outer Loadings")
            outer_loadings_pivoted = loadings.pivot_table(index=loadings.index, columns='Variabel', values='loading')
            all_lvs = sorted(list(set(mv_map.values())))
            all_indicators = sorted(list(mv_map.keys()))
            outer_loadings_pivoted = outer_loadings_pivoted.reindex(index=all_indicators, columns=all_lvs)

            def style_primary_loading(val):
                if pd.isna(val): return ''
                return 'background-color: yellow; color: black;' if val >= 0.708 else 'background-color: red; color: black;'
            st.dataframe(outer_loadings_pivoted.round(3).style.applymap(style_primary_loading).format("{:.3f}", na_rep=""), use_container_width=True)

            st.markdown("### VCross-Loadings")
            all_data_for_corr = pd.concat([df[all_indicators], scores], axis=1)
            correlations = all_data_for_corr.corr().loc[all_indicators, all_lvs]
            correlations.to_csv('cross_loadings.csv')

            def style_crossloadings(row):
                lv_of_indicator = mv_map.get(row.name)
                styles = [''] * len(row)
                if lv_of_indicator and lv_of_indicator in row.index:
                    primary_loading_val = row[lv_of_indicator]
                    max_loading_in_row = row.abs().max()
                    if abs(primary_loading_val) >= max_loading_in_row - 1e-9:
                        lv_col_idx = row.index.get_loc(lv_of_indicator)
                        styles[lv_col_idx] = 'background-color: yellow; color: black;'
                return styles
            st.dataframe(correlations.round(3).style.apply(style_crossloadings, axis=1).format("{:.3f}"), use_container_width=True)

            st.markdown("###  Fornell-Larcker Criterion")            
            lvs_order = ['BI', 'EE', 'FC', 'H', 'HM', 'PE', 'PR', 'PV', 'SI', 'UB']
            lvs_present_in_order = [lv for lv in lvs_order if lv in fornell_larcker_matrix.index]
            fornell_larcker_ordered = fornell_larcker_matrix.reindex(index=lvs_present_in_order, columns=lvs_present_in_order)
            
            def style_fornell_larcker(df):
                styled_df = pd.DataFrame('', index=df.index, columns=df.columns)
                for i, r in df.iterrows():
                    for j, v in r.items():
                        if pd.notna(v):
                            if i == j: 
                                styled_df.loc[i, j] = 'background-color: yellow; color: black; font-weight: bold;'
                            else:
                                if abs(v) > df.loc[j, j]:
                                    styled_df.loc[i, j] = 'background-color: #f8d7da; color: black;'
                                if abs(v) > df.loc[i, i]:
                                    styled_df.loc[i, j] = 'background-color: #f8d7da; color: black;'
                return styled_df

            st.dataframe(fornell_larcker_ordered.round(3).style.apply(style_fornell_larcker, axis=None).format("{:.3f}", na_rep=""), use_container_width=True)

            st.markdown("### Validitas Diskriminan (HTMT Ratio)")
            st.dataframe(htmt_matrix.round(3).style.background_gradient(cmap='Reds', vmin=0.85, vmax=1.0).format("{:.3f}", na_rep=""), use_container_width=True)

        with tab2:
            st.markdown("### Kualitas Prediksi (R-Square)")
            r_squared_df = inner_sum['r_squared'].reset_index()
            r_squared_df.columns = ['Variabel Dependen', 'R^2']
            r_squared_df = r_squared_df[r_squared_df['R^2'] > 0.001]
            
            def get_r2_keterangan(r2):
                if r2 >= 0.75: return "Tinggi"
                elif r2 >= 0.50: return "Moderat"
                elif r2 >= 0.25: return "Lemah"
                return "Sangat Lemah"
            
            r_squared_df['Keterangan'] = r_squared_df['R^2'].apply(get_r2_keterangan)
            st.dataframe(r_squared_df.round(3).set_index('Variabel Dependen').style.format({'R^2': '{:.3f}'}))
            
            st.markdown("### Kualitas Prediksi (Q-Square)")
            st.dataframe(indicator_q2_df.round(3).set_index('Variabel').style.format("{:.3f}"))
            
            st.markdown("### Kolineritas Antar Konstruk (VIF)")
            st.dataframe(indicator_vif_df.round(3).set_index('Indikator').style.format("{:.3f}"))

        with tab3:
            st.markdown("### Pengujian Hipotesis")
            cols_candidate = ['Jalur', 'Original Sample (O)', 'Standard Deviation', 't stat.', 'P-Values', 'Keputusan', 'f2']
            cols_show = [c for c in final_results.columns if c in cols_candidate]
            
            def color_res(val):
                background = "#d4edda" if val == "DITERIMA" else "#f8d7da"
                return f'background-color: {background}; color: black;'
            st.dataframe(final_results[cols_show].round(3).style.applymap(color_res, subset=['Keputusan']).format({
                'Original Sample (O)': '{:.3f}', 'Standard Deviation': '{:.3f}', 
                't stat.': '{:.3f}', 'P-Values': '{:.3f}', 'f2': '{:.3f}'
            }, na_rep="-"), use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan sistem: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()

