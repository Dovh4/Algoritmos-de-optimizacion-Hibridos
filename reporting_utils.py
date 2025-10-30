# Guardar como: reporting_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # Importar os

def generate_excel_report(history_data, filename="reporte_experimento_doa.xlsx"):
    """
    Recibe el historial de datos de una corrida y lo guarda en la ruta 'filename'.
    """
    print(f"\nGenerando reporte en Excel: {filename}")
    df_reporte = pd.DataFrame(history_data)
    df_reporte.to_excel(filename, index=False)
    print(f"Reporte '{filename}' guardado.")

def generate_plots(df_reporte, final_results, filename="analisis_doa_wdbc.png"):
    """
    Genera el dashboard de 4 gráficos y lo guarda en la ruta 'filename'.
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
    ax1.set_ylabel('Error kNN (Interno)', color='blue')
    ax1.plot(df_reporte['iteracion'], df_reporte['error_knn'], color='blue', label='Error kNN')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx() # Eje Y secundario
    ax2.set_ylabel('Num. Características', color='green')
    ax2.plot(df_reporte['iteracion'], df_reporte['num_caracteristicas'], color='green', linestyle='--', label='Num. Caract.')
    ax2.tick_params(axis='y', labelcolor='green')
    plt.title("3. Evolución del Error vs. N° Características")
    plt.grid(True)

    # --- Gráfico 4: Rendimiento Final (Barra) ---
    plt.subplot(2, 2, 4)
    nombres = ['kNN (Todas las Características)', f'kNN-DOA ({num_features} Caract.)']
    precisiones = [acc_full, acc_subset]
    barras = plt.bar(nombres, precisiones, color=['gray', 'blue'])
    plt.ylabel("Precisión en Datos de Test")
    plt.title("4. Rendimiento Final (Validación)")
    plt.ylim(0, 1.05)
    # Añadir etiquetas de valor
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

    plt.tight_layout() # Ajusta los gráficos
    plt.savefig(filename) # Guarda en la ruta especificada
    print(f"Gráficos '{filename}' guardados.")
    plt.close() # Cierra la figura para liberar memoria

def run_statistical_analysis(results_list, excel_filename, plot_filename):
    """
    Toma una lista de resultados (de múltiples corridas) y genera
    un reporte estadístico en Excel y un boxplot en las rutas especificadas.
    """
    print("\n--- Análisis Estadístico (Múltiples Corridas) ---")
    
    # 1. Crear DataFrame con los resultados
    df_stats = pd.DataFrame(results_list)
    
    # 2. Calcular estadísticas descriptivas
    stats_desc = df_stats.describe().transpose()
    print("Estadísticas Descriptivas (Resumen de Corridas):")
    print(stats_desc[['mean', 'std', 'min', 'max']])
    
    # 3. Guardar en Excel
    with pd.ExcelWriter(excel_filename) as writer:
        df_stats.to_excel(writer, sheet_name='Resultados_Raw', index_label='Corrida')
        stats_desc.to_excel(writer, sheet_name='Estadisticas_Descriptivas')
    print(f"Reporte estadístico '{excel_filename}' guardado.")
    
    # 4. Generar Boxplots (Gráficos de Cajas)
    plt.figure(figsize=(12, 6))

    # Boxplot para Precisión
    plt.subplot(1, 2, 1)
    df_plot = pd.DataFrame({
        'kNN-Full': df_stats['precision_full'],
        'kNN-DOA': df_stats['precision_doa']
    })
    df_plot.boxplot(grid=True)
    plt.title('Distribución de Precisión')
    plt.ylabel('Precisión en Test')
    
    # Boxplot para Número de Características
    plt.subplot(1, 2, 2)
    df_stats[['num_features_doa']].boxplot(grid=True)
    plt.title('Distribución de N° Características')
    plt.ylabel('N° Características Seleccionadas')
    
    plt.tight_layout()
    plt.savefig(plot_filename) # Guarda en la ruta especificada
    print(f"Gráficos '{plot_filename}' guardados.")
    plt.close() # Cierra la figura