# Guardar como: run_experiment.py
import numpy as np
import pandas as pd
import random 
import os # Importar os para manejar carpetas
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time

# --- 1. Importa tus módulos personalizados ---
from doa_algorithm import run_DOA
# 
# --- IMPORTANTE: Elige tu wrapper ---
#
# from knn_fs_wrapper_sklearn import KnnFitness  # (Opción CPU)
from knn_fs_wrapper_pytorch import KnnFitness  # (Opción PyTorch GPU)
#
from reporting_utils import generate_excel_report, generate_plots, run_statistical_analysis

def run_single_experiment(seed, k_vecinos, alpha, pop_size, max_iter, X_train, y_train, X_test, y_test):
    """
    Ejecuta una sola corrida completa del experimento con una semilla dada
    para garantizar la replicabilidad.
    """
    print(f"\n--- Iniciando Corrida (Seed={seed}) ---")
    
    # 1. Seteo de Semilla para Replicabilidad
    np.random.seed(seed)
    random.seed(seed) 
    # (Si usas PyTorch, él también captura la semilla de numpy)
    
    # 2. Configurar Evaluador
    evaluador = KnnFitness(X_train, y_train, k_vecinos=k_vecinos, alpha=alpha)
    
    # 3. Configurar y Ejecutar DOA
    PARAMETROS_DOA = {
        'func_objetivo': evaluador.evaluate,
        'dim': evaluador.dim,
        'pop_size': pop_size,
        'max_iter': max_iter,
        'lb': -10.0,
        'ub': 10.0
    }
    
    start_time = time.time()
    best_sol_continua, best_fitness, history = run_DOA(**PARAMETROS_DOA)
    end_time = time.time()
    
    print(f"Corrida (Seed={seed}) finalizada en {end_time - start_time:.2f} segundos.")
    
    # 4. Binarización Final (Determinista 0.5) para la mejor solución
    best_sol_binaria = evaluador._binarizar_determinista(best_sol_continua)
    num_features_seleccionadas = np.sum(best_sol_binaria)

    # 5. Validación FINAL (Test Set)
    
    # a. Precisión Full (Todas las características)
    knn_full = KNeighborsClassifier(n_neighbors=k_vecinos)
    knn_full.fit(X_train, y_train)
    acc_full = knn_full.score(X_test, y_test)
    
    # b. Precisión DOA (Características seleccionadas)
    acc_subset = 0.0
    if num_features_seleccionadas > 0:
        X_train_subset = X_train[:, best_sol_binaria == 1]
        X_test_subset = X_test[:, best_sol_binaria == 1]
        
        knn_subset = KNeighborsClassifier(n_neighbors=k_vecinos)
        knn_subset.fit(X_train_subset, y_train)
        acc_subset = knn_subset.score(X_test_subset, y_test)
    
    # 6. Preparar datos de Reporte (Log de iteraciones)
    log_data = []
    for i in range(max_iter):
        sol_cont = history['best_solution_per_iter'][i]
        sol_bin = evaluador._binarizar_determinista(sol_cont)
        error, reduccion_pct, num_feat = evaluador.evaluate_components(sol_bin)
        
        log_data.append({
            'iteracion': i + 1,
            'mejor_fitness': history['fitness_curve'][i],
            'error_knn': error,
            'pct_reduccion': reduccion_pct,
            'num_caracteristicas': num_feat,
            'diversidad_poblacion': history['diversity_curve'][i]
        })
        
    # 7. Preparar resultados finales (para estadísticas)
    final_results_summary = {
        'seed': seed,
        'precision_full': acc_full,
        'precision_doa': acc_subset,
        'num_features_doa': num_features_seleccionadas,
        'best_fitness_final': best_fitness,
        'tiempo_ejecucion': end_time - start_time
    }
    
    # Paquete de gráficos (para la corrida individual)
    plot_results_package = {
        'acc_full': acc_full,
        'acc_subset': acc_subset,
        'num_features': num_features_seleccionadas
    }
    
    return final_results_summary, log_data, plot_results_package


# --- INICIO DEL SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    
    # --- A. Definición de Carpetas de Resultados ---
    RESULTS_DIR = "results"
    STATS_DIR = os.path.join(RESULTS_DIR, "statistical_summary")
    LOGS_DIR = os.path.join(RESULTS_DIR, "detailed_run_logs")
    
    # Crear carpetas si no existen
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # --- B. Carga de Datos (Una sola vez) ---
    print("Cargando y preparando datos (WDBC)...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # --- C. Parámetros del Experimento ---
    K_VECINOS = 7
    ALPHA = 0.9       
    POP_SIZE = 30     
    MAX_ITER = 100    
    NUM_RUNS = 30     
    
    # --- D. Bucle de Múltiples Corridas Estadísticas ---
    resultados_estadisticos = []
    
    for i in range(NUM_RUNS):
        final_results, log_data, plot_results = run_single_experiment(
            seed=i,
            k_vecinos=K_VECINOS, 
            alpha=ALPHA, 
            pop_size=POP_SIZE, 
            max_iter=MAX_ITER,
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test
        )
        
        resultados_estadisticos.append(final_results)
        
        # --- E. Reporte de la *Primera* Corrida (seed=0) ---
        if i == 0:
            print("\n--- Generando Reportes Detallados (Solo para la primera corrida) ---")
            
            # Definir rutas de archivo
            log_excel_path = os.path.join(LOGS_DIR, "reporte_corrida_seed_0.xlsx")
            log_plot_path = os.path.join(LOGS_DIR, "graficos_corrida_seed_0.png")
            
            # Generar reportes
            generate_excel_report(log_data, filename=log_excel_path)
            generate_plots(pd.DataFrame(log_data), plot_results, filename=log_plot_path)

    # --- F. Análisis Estadístico Final ---
    
    # Definir rutas de archivo
    stats_excel_path = os.path.join(STATS_DIR, "reporte_estadistico_final.xlsx")
    stats_plot_path = os.path.join(STATS_DIR, "analisis_estadistico_boxplots.png")
    
    # Generar reportes
    run_statistical_analysis(
        resultados_estadisticos, 
        excel_filename=stats_excel_path, 
        plot_filename=stats_plot_path
    )

    print("\n--- Experimento Estadístico Completo Finalizado ---")