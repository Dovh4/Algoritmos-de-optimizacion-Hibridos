# Guardar como: best_solution_finder.py
import pandas as pd
import os
import sys

# --- 1. Definición de Rutas de Archivos ---
OUTPUT_DIR = "global_best_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_OUTPUT = os.path.join(OUTPUT_DIR, "best_global_solutions.csv")

# --- Rutas de WDBC ---
DOA_WDBC_PATH = os.path.join("results_wdbc_DOA", "statistical_summary", "reporte_estadistico_final_wdbc.csv")
PSO_WDBC_PATH = os.path.join("results_wdbc_PSO", "statistical_summary", "reporte_estadistico_final_wdbc.csv")
GWO_WDBC_PATH = os.path.join("results_wdbc_GWO", "statistical_summary", "reporte_estadistico_final_wdbc.csv") # NUEVO

# --- Rutas de Madelon ---
DOA_MADELON_PATH = os.path.join("results_madelon_DOA", "statistical_summary", "reporte_estadistico_final_madelon.csv")
PSO_MADELON_PATH = os.path.join("results_madelon_PSO", "statistical_summary", "reporte_estadistico_final_madelon.csv")
GWO_MADELON_PATH = os.path.join("results_madelon_GWO", "statistical_summary", "reporte_estadistico_final_madelon.csv") # NUEVO

def find_best_solution(csv_path, algorithm_name, dataset_name):
    """
    Carga los resultados, encuentra la corrida con el F1-Score más alto (la mejor solución global)
    y retorna TODAS sus métricas.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Archivo no encontrado en {csv_path}. Omitiendo {algorithm_name}-{dataset_name}.", file=sys.stderr)
        return None
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error al leer el archivo {csv_path}: {e}", file=sys.stderr)
        return None

    # Clave de la métrica (ej: 'f1_doa', 'f1_pso', 'f1_gwo')
    metric_key = f'f1_{algorithm_name.lower()}'
    
    if metric_key not in df.columns:
        print(f"Error: Clave '{metric_key}' no encontrada en {csv_path}. Columnas disponibles: {df.columns}", file=sys.stderr)
        return None

    idx_best = df[metric_key].idxmax() 
    best_run = df.loc[idx_best].to_dict()

    # Preparar el diccionario de resultados para el reporte final
    report_data = {
        'Dataset': dataset_name,
        'Algoritmo': algorithm_name.upper(), # Guardar como DOA, PSO, GWO
        'Corrida_ID': best_run['seed'],
        
        # --- Métricas de la Solución Óptima (Metaheurística) ---
        'F1_Optimizado': best_run[f'f1_{algorithm_name.lower()}'],
        'Precision_Optimizado': best_run[f'precision_{algorithm_name.lower()}'],
        'Recall_Optimizado': best_run[f'recall_{algorithm_name.lower()}'],
        'Num_Features': best_run[f'num_features_{algorithm_name.lower()}'],
        'RD_Dimensionality': best_run[f'rd_{algorithm_name.lower()}'],
        'Mejor_Fitness_Final': best_run['best_fitness_final'],
        'Tiempo_Ejecucion_s': best_run['tiempo_ejecucion'],
        
        # --- Métricas del Baseline (KNN-Full) ---
        'F1_KNN_Full': best_run['f1_full'],
        'Precision_KNN_Full': best_run['precision_full'],
        'Recall_KNN_Full': best_run['recall_full'],

        # --- Vector Binario ---
        'Vector_Binario_Completo': best_run['best_binary_vector'] 
    }
    return report_data

if __name__ == "__main__":
    
    print("--- Iniciando Búsqueda de la Mejor Solución Global ---")
    all_best_results = []
    
    # --- Ejecutar la búsqueda para los 6 escenarios ---
    
    # WDBC
    all_best_results.append(find_best_solution(DOA_WDBC_PATH, 'doa', 'WDBC'))
    all_best_results.append(find_best_solution(PSO_WDBC_PATH, 'pso', 'WDBC'))
    all_best_results.append(find_best_solution(GWO_WDBC_PATH, 'gwo', 'WDBC')) # NUEVO
    
    # Madelon
    all_best_results.append(find_best_solution(DOA_MADELON_PATH, 'doa', 'Madelon'))
    all_best_results.append(find_best_solution(PSO_MADELON_PATH, 'pso', 'Madelon'))
    all_best_results.append(find_best_solution(GWO_MADELON_PATH, 'gwo', 'Madelon')) # NUEVO

    # --- Generar Reporte Final (Filtrando resultados None) ---
    all_best_results = [res for res in all_best_results if res is not None]

    if all_best_results:
        df_reporte_global = pd.DataFrame(all_best_results)
        df_reporte_global.to_csv(REPORT_OUTPUT, index=False)
        
        print(f"\n--- ✅ Reporte de Mejores Soluciones Generado ---")
        print(f"Archivo guardado en: {REPORT_OUTPUT}")
        
        print("\nResumen de las métricas clave del mejor resultado:")
        print(df_reporte_global[['Dataset', 'Algoritmo', 'F1_Optimizado', 'Num_Features', 'Tiempo_Ejecucion_s', 'Corrida_ID']])