# Guardar como: reporting_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

def generate_csv_report(history_data, filename="reporte_experimento_doa.csv"):
    """
    Recibe el historial de datos de una corrida y lo guarda en la ruta 'filename' (CSV).
    """
    print(f"\nGenerando reporte en CSV: {filename}")
    df_reporte = pd.DataFrame(history_data)
    df_reporte.to_csv(filename, index=False)
    print(f"Reporte '{filename}' guardado.")

def generate_plots(df_reporte, final_results, filename, algoritmo):
    """
    Genera el dashboard de 4 gráficos y lo guarda en la ruta 'filename'.
    (Esta función no requiere cambios)
    """
    print(f"Generando gráficos de análisis: {filename}")
    
    # Extraer resultados finales para el gráfico
    acc_full = final_results.get('acc_full', 0)
    acc_subset = final_results.get('acc_subset', 0)
    num_features = final_results.get('num_features', 0)
    
    plt.figure(figsize=(20, 12))

    # --- Gráfico 1: Convergencia del Fitness ---
    plt.subplot(2, 2, 1)
    plt.plot(df_reporte['iteracion'], df_reporte['mejor_fitness'])
    plt.title("1. Convergencia del Fitness (Global)")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor Fitness")
    plt.grid(True)

    # --- Gráfico 2: Exploración vs Explotación ---
    plt.subplot(2, 2, 2)
    plt.plot(df_reporte['iteracion'], df_reporte['diversidad_poblacion'])
    plt.title("2. Exploración vs. Explotación (Diversidad)")
    plt.xlabel("Iteración")
    plt.ylabel("Diversidad de la Población (StdDev Media)")
    plt.grid(True)

    # --- Gráfico 3: Rendimiento (Error y Características) ---
    ax1 = plt.subplot(2, 2, 3)
    ax1.set_xlabel('Iteración')
    # (Ajuste menor para Madelon, si el log usa 'penalizacion_tamano')
    y_label_error = 'Error kNN (Interno)'
    y_data_error = df_reporte.get('error_knn', pd.Series(dtype='float'))
    
    ax1.set_ylabel(y_label_error, color='blue')
    ax1.plot(df_reporte['iteracion'], y_data_error, color='blue', label='Error kNN')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx() # Eje Y secundario
    ax2.set_ylabel('Num. Características', color='green')
    ax2.plot(df_reporte['iteracion'], df_reporte['num_caracteristicas'], color='green', linestyle='--', label='Num. Caract.')
    ax2.tick_params(axis='y', labelcolor='green')
    plt.title("3. Evolución del Error vs. N° Características")
    plt.grid(True)

    # --- Gráfico 4: Rendimiento Final (Barra) ---
    plt.subplot(2, 2, 4)
    nombres = ['kNN (Todas las Características)', f'kNN-{algoritmo} ({num_features} Caract.)']
    precisiones = [acc_full, acc_subset]
    barras = plt.bar(nombres, precisiones, color=['gray', 'blue'])
    plt.ylabel("Precisión en Datos de Test")
    plt.title("4. Rendimiento Final (Validación)")
    plt.ylim(0, 1.05)
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

    plt.tight_layout() 
    plt.savefig(filename) 
    print(f"Gráficos '{filename}' guardados.")
    plt.close() 

def run_statistical_analysis(results_list, csv_filename, plot_filename, algoritmo):
    """
    Toma una lista de resultados (de múltiples corridas) y genera
    un reporte estadístico en CSV y un boxplot en las rutas especificadas.
    """
    print("\n--- Análisis Estadístico (Múltiples Corridas) ---")
    
    # 1. Crear DataFrame con los resultados
    df_stats = pd.DataFrame(results_list)
    
    # 2. Calcular estadísticas descriptivas
    stats_desc = df_stats.describe().transpose()
    print("Estadísticas Descriptivas (Resumen de Corridas):")
    print(stats_desc[['mean', 'std', 'min', 'max']])
    
    # 3. Guardar en CSV (MODIFICADO)
    
    # a. Guardar los resultados crudos (raw) de las 30 corridas
    df_stats.to_csv(csv_filename, index_label='Corrida')
    print(f"Reporte estadístico (Raw) '{csv_filename}' guardado.")
    
    # b. Guardar las estadísticas descriptivas (mean, std, etc.)
    desc_filename = csv_filename.replace('.csv', '_descriptivas.csv')
    stats_desc.to_csv(desc_filename)
    print(f"Reporte estadístico (Descriptivo) '{desc_filename}' guardado.")
    
    # 4. Generar Boxplots (Gráficos de Cajas)
    plt.figure(figsize=(18, 6)) # Ancho aumentado

    # Boxplot para Precisión
    plt.subplot(1, 3, 1)
    df_plot_acc = pd.DataFrame({
        'kNN-Full (Acc)': df_stats['precision_full'],
        f'kNN-{algoritmo} (Acc)': df_stats[f'precision_{algoritmo}']
    })
    df_plot_acc.boxplot(grid=True)
    plt.title('Distribución de Precisión (Acc)')
    plt.ylabel('Precisión en Test')
    
    # Boxplot para F1-Score (NUEVO)
    plt.subplot(1, 3, 2)
    df_plot_f1 = pd.DataFrame({
        'kNN-Full (F1)': df_stats['f1_full'],
        f'kNN-{algoritmo} (F1)': df_stats[f'f1_{algoritmo}']
    })
    df_plot_f1.boxplot(grid=True)
    plt.title('Distribución de F1-Score')
    plt.ylabel('F1-Score en Test')

    # Boxplot para Número de Características
    plt.subplot(1, 3, 3)
    df_stats[[f'num_features_{algoritmo}']].boxplot(grid=True)
    plt.title(f'Distribución de N° Características ({algoritmo})')
    plt.ylabel('N° Características Seleccionadas')
    
    plt.tight_layout()
    plt.savefig(plot_filename) # Guarda en la ruta especificada
    print(f"Gráficos '{plot_filename}' guardados.")
    plt.close() # Cierra la figura