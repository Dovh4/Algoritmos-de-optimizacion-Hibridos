# Guardar como: comparison_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- 1. Definición de Rutas de Archivos ---
OUTPUT_DIR = "comparative_analysis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Rutas de WDBC ---
DOA_WDBC_PATH = os.path.join("results_wdbc_DOA", "statistical_summary", "reporte_estadistico_final_wdbc.csv")
PSO_WDBC_PATH = os.path.join("results_wdbc_PSO", "statistical_summary", "reporte_estadistico_final_wdbc.csv")
GWO_WDBC_PATH = os.path.join("results_wdbc_GWO", "statistical_summary", "reporte_estadistico_final_wdbc.csv") # NUEVO
PLOT_WDBC_BASENAME = os.path.join(OUTPUT_DIR, "analisis_comparativo_WDBC") # Basename para plots

# --- Rutas de Madelon ---
DOA_MADELON_PATH = os.path.join("results_madelon_DOA", "statistical_summary", "reporte_estadistico_final_madelon.csv")
PSO_MADELON_PATH = os.path.join("results_madelon_PSO", "statistical_summary", "reporte_estadistico_final_madelon.csv")
GWO_MADELON_PATH = os.path.join("results_madelon_GWO", "statistical_summary", "reporte_estadistico_final_madelon.csv") # NUEVO
PLOT_MADELON_BASENAME = os.path.join(OUTPUT_DIR, "analisis_comparativo_MADELON") # Basename para plots


def load_data(csv_path):
    """Carga el CSV de resultados estadísticos."""
    if not os.path.exists(csv_path):
        print(f"Error: Archivo no encontrado en {csv_path}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error al leer el archivo {csv_path}: {e}", file=sys.stderr)
        return None

def plot_comparison_boxplots(df_doa, df_pso, df_gwo, output_basename, dataset_name):
    """
    Genera boxplots comparativos para Precisión, F1 y N° Características.
    AHORA GENERA 3 GRÁFICOS SEPARADOS.
    """
    print(f"Generando gráficos comparativos para {dataset_name}...")
    
    # --- Gráfico 1: Precisión (Accuracy) ---
    plt.figure(figsize=(10, 7))
    plt.title(f'Distribución de Precisión (Accuracy) - {dataset_name}', fontsize=16)
    
    df_plot_acc = pd.DataFrame({
        'KNN-Full': df_doa['precision_full'], # Baseline
        'DOA-KNN': df_doa['precision_doa'],
        'PSO-KNN': df_pso['precision_pso'],
        'GWO-KNN': df_gwo['precision_gwo']  # NUEVO
    })
    
    df_plot_acc.boxplot(grid=True, patch_artist=True)
    plt.ylabel('Precisión en Test')
    plt.ylim(0.5, 1.05) # Ajustar Y-axis si es necesario
    
    plot_filename_acc = f"{output_basename}_Accuracy.png"
    plt.savefig(plot_filename_acc)
    plt.close()
    print(f"Gráfico de Precisión guardado en: {plot_filename_acc}")

    # --- Gráfico 2: F1-Score ---
    plt.figure(figsize=(10, 7))
    plt.title(f'Distribución de F1-Score - {dataset_name}', fontsize=16)

    df_plot_f1 = pd.DataFrame({
        'KNN-Full': df_doa['f1_full'], # Baseline
        'DOA-KNN': df_doa['f1_doa'],
        'PSO-KNN': df_pso['f1_pso'],
        'GWO-KNN': df_gwo['f1_gwo']  # NUEVO
    })
    
    df_plot_f1.boxplot(grid=True, patch_artist=True)
    plt.ylabel('F1-Score en Test')
    plt.ylim(0.5, 1.05) # Ajustar Y-axis si es necesario

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
        'GWO-KNN': df_gwo['num_features_gwo'] # NUEVO
    })
    
    df_plot_feat.boxplot(grid=True, patch_artist=True)
    plt.ylabel('N° Características')

    plot_filename_feat = f"{output_basename}_Num_Features.png"
    plt.savefig(plot_filename_feat)
    plt.close()
    print(f"Gráfico de N° Características guardado en: {plot_filename_feat}\n")


def run_analysis(doa_csv, pso_csv, gwo_csv, plot_basename, dataset_name):
    """
    Función principal para un dataset: carga datos y genera el gráfico.
    """
    df_doa = load_data(doa_csv)
    df_pso = load_data(pso_csv)
    df_gwo = load_data(gwo_csv) # NUEVO

    if df_doa is not None and df_pso is not None and df_gwo is not None:
        # (Se podrían agregar más chequeos de longitud)
        
        plot_comparison_boxplots(df_doa, df_pso, df_gwo, plot_basename, dataset_name)
    else:
        print(f"No se pudo generar el análisis para {dataset_name} debido a archivos faltantes.")


# --- Ejecución Principal ---
if __name__ == "__main__":
    
    print("--- Iniciando Análisis Comparativo ---")
    
    # --- Generar Gráficos para WDBC ---
    run_analysis(
        doa_csv=DOA_WDBC_PATH,
        pso_csv=PSO_WDBC_PATH,
        gwo_csv=GWO_WDBC_PATH, # NUEVO
        plot_basename=PLOT_WDBC_BASENAME,
        dataset_name="WDBC"
    )
    
    # --- Generar Gráficos para Madelon ---
    run_analysis(
        doa_csv=DOA_MADELON_PATH,
        pso_csv=PSO_MADELON_PATH,
        gwo_csv=GWO_MADELON_PATH, # NUEVO
        plot_basename=PLOT_MADELON_BASENAME,
        dataset_name="Madelon"
    )
    
    print("--- Análisis Comparativo Finalizado ---")