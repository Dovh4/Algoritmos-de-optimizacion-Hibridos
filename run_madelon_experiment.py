# Guardar como: run_madelon_experiment.py
import numpy as np
import pandas as pd
import random 
import os 
import time
import torch

# (Importaciones sin cambios)
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier 
from doa_algorithm import run_DOA
from knn_fs_wrapper_pytorch import KnnFitness
from reporting_utils import generate_excel_report, generate_plots, run_statistical_analysis

def run_single_experiment(seed, k_vecinos, alpha, pop_size, max_iter, 
                          search_lb, search_ub, init_lb, init_ub, 
                          X_train, y_train, X_test, y_test):
    """
    Ejecuta una sola corrida completa del experimento con una semilla dada.
    """
    print(f"\n--- Iniciando Corrida (Seed={seed}) ---")
    
    np.random.seed(seed)
    random.seed(seed) 

    # *** CONFIGURACIÓN PARA PYTORCH/GPU (DETERMINISMO) ***
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    # ******************************************************
    
    evaluador = KnnFitness(X_train, y_train, k_vecinos=k_vecinos, alpha=alpha)
    
    # Define los parámetros para el DOA
    PARAMETROS_DOA = {
        'func_objetivo': evaluador.evaluate,
        'dim': evaluador.dim,
        'pop_size': pop_size,
        'max_iter': max_iter,
        'lb': search_lb,      # Límites de Búsqueda (Clipping)
        'ub': search_ub,
        'init_lb': init_lb,   # Límites de Inicialización
        'init_ub': init_ub
    }
    
    # Inicia la optimización
    start_time = time.time()
    best_sol_continua, best_fitness, history = run_DOA(**PARAMETROS_DOA)
    end_time = time.time()
    
    print(f"Corrida (Seed={seed}) finalizada en {end_time - start_time:.2f} segundos.")
    
    # (El resto de la función run_single_experiment es idéntico)
    best_sol_binaria = evaluador._binarizar_determinista(best_sol_continua)
    num_features_seleccionadas = np.sum(best_sol_binaria)
    knn_full = KNeighborsClassifier(n_neighbors=k_vecinos)
    knn_full.fit(X_train, y_train)
    acc_full = knn_full.score(X_test, y_test)
    acc_subset = 0.0
    if num_features_seleccionadas > 0:
        X_train_subset = X_train[:, best_sol_binaria == 1]
        X_test_subset = X_test[:, best_sol_binaria == 1]
        knn_subset = KNeighborsClassifier(n_neighbors=k_vecinos)
        knn_subset.fit(X_train_subset, y_train)
        acc_subset = knn_subset.score(X_test_subset, y_test)
    log_data = []
    for i in range(max_iter):
        sol_cont = history['best_solution_per_iter'][i]
        sol_bin = evaluador._binarizar_determinista(sol_cont)
        error, reduccion_pct, num_feat = evaluador.evaluate_components(sol_bin)
        log_data.append({
            'iteracion': i + 1,
            'mejor_fitness': history['fitness_curve'][i],
            'error_knn': error,
            'penalizacion_tamano': reduccion_pct,
            'num_caracteristicas': num_feat,
            'diversidad_poblacion': history['diversity_curve'][i]
        })
    final_results_summary = {
        'seed': seed,
        'precision_full': acc_full,
        'precision_doa': acc_subset,
        'num_features_doa': num_features_seleccionadas,
        'best_fitness_final': best_fitness,
        'tiempo_ejecucion': end_time - start_time
    }
    plot_results_package = {
        'acc_full': acc_full,
        'acc_subset': acc_subset,
        'num_features': num_features_seleccionadas
    }
    return final_results_summary, log_data, plot_results_package


# --- INICIO DEL SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    
    # (Definición de carpetas y carga de datos sin cambios)
    RESULTS_DIR = "results_madelon"
    STATS_DIR = os.path.join(RESULTS_DIR, "statistical_summary")
    LOGS_DIR = os.path.join(RESULTS_DIR, "detailed_run_logs")
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    print("Cargando y preparando datos (Madelon)...")
    madelon = fetch_openml(name='madelon', version=1, parser='auto')
    X = madelon.data.values
    y = madelon.target.values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Dataset Madelon cargado. Dimensiones: {X.shape}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # --- C. Parámetros del Experimento (Intento 7: Inicialización Sesgada Correcta) ---
    K_VECINOS = 20     # (k=15 para señal suave)
    ALPHA = 0.9        # (Balance 50% Error, 50% Imán)
    
    POP_SIZE = 150     # (Población grande)
    MAX_ITER = 300     # (Iteraciones altas)
    NUM_RUNS = 30      
    
    # --- LÍMITES DE BÚSQUEDA (Clipping) ---
    SEARCH_LB = -10.0
    SEARCH_UB = 10.0
    
    # --- LÍMITES DE INICIALIZACIÓN (Prob 0,7% - 73%) ---
    INIT_LB = -5.0     
    INIT_UB = +1.0     
    
    
    # Bucle de Múltiples Corridas Estadísticas
    resultados_estadisticos = []
    
    print(f"Iniciando {NUM_RUNS} corridas para Madelon (Dim={X.shape[1]})...")
    print(f"Estrategia: Inicialización sesgada [{INIT_LB}, {INIT_UB}], Búsqueda [{SEARCH_LB}, {SEARCH_UB}]")

    for i in range(NUM_RUNS):
        final_results, log_data, plot_results = run_single_experiment(
            seed=i,
            k_vecinos=K_VECINOS, 
            alpha=ALPHA, 
            pop_size=POP_SIZE, 
            max_iter=MAX_ITER,
            search_lb=SEARCH_LB, # Pasar los límites de búsqueda
            search_ub=SEARCH_UB,
            init_lb=INIT_LB,     # Pasar los límites de inicialización
            init_ub=INIT_UB,
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test
        )
        
        resultados_estadisticos.append(final_results)
        
        # Reporte de la *Primera* Corrida (seed=0)
        if i == 0:
            print("\n--- Generando Reportes Detallados (Solo para la primera corrida) ---")
            log_excel_path = os.path.join(LOGS_DIR, "reporte_madelon_seed_0.xlsx")
            log_plot_path = os.path.join(LOGS_DIR, "graficos_madelon_seed_0.png")
            generate_excel_report(log_data, filename=log_excel_path)
            generate_plots(pd.DataFrame(log_data), plot_results, filename=log_plot_path)

    # Análisis Estadístico Final
    stats_excel_path = os.path.join(STATS_DIR, "reporte_estadistico_final_madelon.xlsx")
    stats_plot_path = os.path.join(STATS_DIR, "analisis_estadistico_boxplots_madelon.png")
    run_statistical_analysis(
        resultados_estadisticos, 
        excel_filename=stats_excel_path, 
        plot_filename=stats_plot_path
    )

    print("\n--- Experimento Estadístico (Madelon) Completo Finalizado ---")