# Guardar como: comparison_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
# --- NUEVA IMPORTACIÓN ---
from scipy.stats import wilcoxon

# --- 1. Definición de Rutas de Archivos ---
OUTPUT_DIR = "comparative_analysis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (Rutas de WDBC y Madelon sin cambios)
# --- Rutas de WDBC ---
DOA_WDBC_PATH = os.path.join("results_wdbc_DOA", "statistical_summary", "reporte_estadistico_final_wdbc.csv")
PSO_WDBC_PATH = os.path.join("results_wdbc_PSO", "statistical_summary", "reporte_estadistico_final_wdbc.csv")
GWO_WDBC_PATH = os.path.join("results_wdbc_GWO", "statistical_summary", "reporte_estadistico_final_wdbc.csv")
PLOT_WDBC_BASENAME = os.path.join(OUTPUT_DIR, "analisis_comparativo_WDBC")

# --- Rutas de Madelon ---
DOA_MADELON_PATH = os.path.join("results_madelon_DOA", "statistical_summary", "reporte_estadistico_final_madelon.csv")
PSO_MADELON_PATH = os.path.join("results_madelon_PSO", "statistical_summary", "reporte_estadistico_final_madelon.csv")
GWO_MADELON_PATH = os.path.join("results_madelon_GWO", "statistical_summary", "reporte_estadistico_final_madelon.csv")
PLOT_MADELON_BASENAME = os.path.join(OUTPUT_DIR, "analisis_comparativo_MADELON")


def load_data(csv_path):
    # (Función sin cambios)
    if not os.path.exists(csv_path):
        print(f"Error: Archivo no encontrado en {csv_path}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error al leer el archivo {csv_path}: {e}", file=sys.stderr)
        return None

def summarize_stats(df, alg_name):
    # (Función sin cambios)
    try:
        stats = {
            'Algorithm': f"{alg_name.upper()}-KNN",
            'Accuracy (Mean)': df[f'precision_{alg_name}'].mean(),
            'Accuracy (Std)': df[f'precision_{alg_name}'].std(),
            'F1-Score (Mean)': df[f'f1_{alg_name}'].mean(),
            'F1-Score (Std)': df[f'f1_{alg_name}'].std(),
            'N° Features (Mean)': df[f'num_features_{alg_name}'].mean(),
            'N° Features (Std)': df[f'num_features_{alg_name}'].std(),
            'Time (Mean)': df['tiempo_ejecucion'].mean(),
            'Time (Std)': df['tiempo_ejecucion'].std()
        }
        return stats
    except KeyError as e:
        print(f"Error al sumarizar {alg_name}: Clave no encontrada {e}", file=sys.stderr)
        return None

def get_full_stats(df, total_features):
    # (Función sin cambios)
    stats = {
        'Algorithm': "KNN-Full",
        'Accuracy (Mean)': df['precision_full'].mean(),
        'Accuracy (Std)': df['precision_full'].std(),
        'F1-Score (Mean)': df['f1_full'].mean(),
        'F1-Score (Std)': df['f1_full'].std(),
        'N° Features (Mean)': total_features,
        'N° Features (Std)': 0.0,
        'Time (Mean)': pd.NA,
        'Time (Std)': pd.NA
    }
    return stats

def plot_comparison_boxplots(df_doa, df_pso, df_gwo, output_basename, dataset_name):
    # (Función sin cambios)
    print(f"Generando gráficos comparativos para {dataset_name}...")
    
    # --- Gráfico 1: Precisión (Accuracy) ---
    plt.figure(figsize=(10, 7))
    plt.title(f'Distribución de Precisión (Accuracy) - {dataset_name}', fontsize=16)
    df_plot_acc = pd.DataFrame({
        'KNN-Full': df_doa['precision_full'],
        'DOA-KNN': df_doa['precision_doa'],
        'PSO-KNN': df_pso['precision_pso'],
        'GWO-KNN': df_gwo['precision_gwo']
    })
    df_plot_acc.boxplot(grid=True, patch_artist=True)
    plt.ylabel('Precisión en Test')
    plt.ylim(0.5, 1.05)
    plot_filename_acc = f"{output_basename}_Accuracy.png"
    plt.savefig(plot_filename_acc)
    plt.close()
    print(f"Gráfico de Precisión guardado en: {plot_filename_acc}")

    # (Gráficos F1-Score y Num_Features sin cambios)
    # --- Gráfico 2: F1-Score ---
    plt.figure(figsize=(10, 7))
    plt.title(f'Distribución de F1-Score - {dataset_name}', fontsize=16)
    df_plot_f1 = pd.DataFrame({
        'KNN-Full': df_doa['f1_full'],
        'DOA-KNN': df_doa['f1_doa'],
        'PSO-KNN': df_pso['f1_pso'],
        'GWO-KNN': df_gwo['f1_gwo']
    })
    df_plot_f1.boxplot(grid=True, patch_artist=True)
    plt.ylabel('F1-Score en Test')
    plt.ylim(0.5, 1.05)
    plot_filename_f1 = f"{output_basename}_F1_Score.png"
    plt.savefig(plot_filename_f1)
    plt.close()
    print(f"Gráfico de F1-Score guardado en: {plot_filename_f1}")

    # --- Gráfico 3: Número de Características ---
    plt.figure(figsize=(10, 7))
    plt.title(f'Distribución de N° Características Seleccionadas - {dataset_name}', fontsize=16)
    df_plot_feat = pd.DataFrame({
        'DOA-KNN': df_doa['num_features_doa'],
        'PSO-KNN': df_pso['num_features_pso'],
        'GWO-KNN': df_gwo['num_features_gwo']
    })
    df_plot_feat.boxplot(grid=True, patch_artist=True)
    plt.ylabel('N° Características')
    plot_filename_feat = f"{output_basename}_Num_Features.png"
    plt.savefig(plot_filename_feat)
    plt.close()
    print(f"Gráfico de N° Características guardado en: {plot_filename_feat}\n")

# --- NUEVA FUNCIÓN ---
def run_wilcoxon_analysis(df_doa, df_pso, df_gwo):
    """
    Ejecuta el Test de Wilcoxon Signed-Rank en las 30 corridas
    para el dataset Madelon (el más crítico).
    """
    print("\n--- Análisis de Significancia Estadística (Test de Wilcoxon) - Madelon ---")
    
    # Lista para guardar los resultados
    wilcoxon_results = []
    
    # Comparaciones de F1-Score (Métrica clave)
    # Hacemos el test sobre la diferencia (DOA > Baseline)
    try:
        # 1. DOA vs PSO (F1-Score)
        stat_f1_pso, p_f1_pso = wilcoxon(df_doa['f1_doa'], df_pso['f1_pso'], alternative='greater')
        wilcoxon_results.append(['DOA-KNN vs. PSO-KNN', 'F1-Score', p_f1_pso, p_f1_pso < 0.05])
        
        # 2. DOA vs GWO (F1-Score)
        stat_f1_gwo, p_f1_gwo = wilcoxon(df_doa['f1_doa'], df_gwo['f1_gwo'], alternative='greater')
        wilcoxon_results.append(['DOA-KNN vs. GWO-KNN', 'F1-Score', p_f1_gwo, p_f1_gwo < 0.05])
        
        # Comparaciones de N° Features (Métrica secundaria)
        # Hacemos el test sobre la diferencia (DOA < Baseline)
        
        # 3. DOA vs PSO (N° Features)
        stat_n_pso, p_n_pso = wilcoxon(df_doa['num_features_doa'], df_pso['num_features_pso'], alternative='less')
        wilcoxon_results.append(['DOA-KNN vs. PSO-KNN', 'N° Features', p_n_pso, p_n_pso < 0.05])

        # 4. DOA vs GWO (N° Features)
        stat_n_gwo, p_n_gwo = wilcoxon(df_doa['num_features_doa'], df_gwo['num_features_gwo'], alternative='less')
        wilcoxon_results.append(['DOA-KNN vs. GWO-KNN', 'N° Features', p_n_gwo, p_n_gwo < 0.05])
        
        # Crear DataFrame para imprimir
        df_wilcoxon = pd.DataFrame(wilcoxon_results, columns=['Comparación', 'Métrica', 'p-value', 'Significativo (p < 0.05)'])
        
        print(df_wilcoxon.to_markdown(index=False, floatfmt=".4e")) # Imprimir en notación científica
        
    except Exception as e:
        print(f"Error durante el Test de Wilcoxon: {e}", file=sys.stderr)
        print("Asegúrese de que los archivos CSV de Madelon (DOA, PSO, GWO) no estén vacíos y contengan 30 corridas.")


# --- FUNCIÓN run_analysis (MODIFICADA) ---
def run_analysis(doa_csv, pso_csv, gwo_csv, plot_basename, dataset_name, total_features):
    """
    Función principal para un dataset: carga datos, genera reportes CSV y gráficos.
    """
    df_doa = load_data(doa_csv)
    df_pso = load_data(pso_csv)
    df_gwo = load_data(gwo_csv)

    if df_doa is not None and df_pso is not None and df_gwo is not None:
        
        # 1. Generar Reporte Estadístico (Mean/Std)
        results_list = []
        results_list.append(summarize_stats(df_doa, 'doa'))
        results_list.append(summarize_stats(df_pso, 'pso'))
        results_list.append(summarize_stats(df_gwo, 'gwo'))
        results_list.append(get_full_stats(df_doa, total_features))
        
        results_list = [res for res in results_list if res is not None]
        
        if results_list:
            df_summary = pd.DataFrame(results_list).set_index('Algorithm')
            print(f"\n--- Resumen Estadístico: {dataset_name} (Mean & Std de 30 Corridas) ---")
            print(df_summary.to_markdown(floatfmt=".4f"))
            
            summary_csv_path = f"{plot_basename}_summary_stats.csv"
            df_summary.to_csv(summary_csv_path)
            print(f"Resumen estadístico guardado en: {summary_csv_path}\n")

        # 2. Generar Gráficos
        if len(df_doa) != len(df_pso) or len(df_doa) != len(df_gwo):
            print(f"Advertencia: Los archivos de {dataset_name} tienen diferente número de corridas.")
            
        plot_comparison_boxplots(df_doa, df_pso, df_gwo, plot_basename, dataset_name)
        
        # --- NUEVO: Ejecutar Wilcoxon SOLO para Madelon ---
        if dataset_name == "Madelon":
            run_wilcoxon_analysis(df_doa, df_pso, df_gwo)
            
    else:
        print(f"No se pudo generar el análisis para {dataset_name} debido a archivos faltantes.")


# --- Ejecución Principal (MODIFICADA) ---
if __name__ == "__main__":
    
    print("--- Iniciando Análisis Comparativo ---")
    
    # --- Generar Gráficos y CSV para WDBC ---
    run_analysis(
        doa_csv=DOA_WDBC_PATH,
        pso_csv=PSO_WDBC_PATH,
        gwo_csv=GWO_WDBC_PATH,
        plot_basename=PLOT_WDBC_BASENAME,
        dataset_name="WDBC",
        total_features=30
    )
    
    # --- Generar Gráficos, CSV y Wilcoxon para Madelon ---
    run_analysis(
        doa_csv=DOA_MADELON_PATH,
        pso_csv=PSO_MADELON_PATH,
        gwo_csv=GWO_MADELON_PATH,
        plot_basename=PLOT_MADELON_BASENAME,
        dataset_name="Madelon",
        total_features=500
    )
    
    print("\n--- Análisis Comparativo Finalizado ---")