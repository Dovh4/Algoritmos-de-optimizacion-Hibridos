# Guardar como: gwo_algorithm.py
import numpy as np
import random

def run_GWO(func_objetivo, dim, pop_size, max_iter, lb, ub, init_lb=None, init_ub=None):
    """
    Ejecuta el Grey Wolf Optimizer (GWO) completo y autocontenido.
    
    Sigue la misma firma y estructura de run_DOA/run_PSO para ser
    intercambiable en los scripts de experimentación.
    """
    
    # --- 0. Manejo de Límites ---
    if init_lb is None: init_lb = lb
    if init_ub is None: init_ub = ub

    # --- 1. Inicialización de la población ---
    population = np.zeros((pop_size, dim))
    for i in range(pop_size):
        population[i] = init_lb + np.random.rand(dim) * (init_ub - init_lb)
    
    fitness = np.full(pop_size, np.inf)
    
    # GWO requiere ordenar para encontrar Alfa, Beta, Delta
    sorted_indices = np.argsort(fitness)
    Xalfa = population[sorted_indices[0]].copy()
    Xbeta = population[sorted_indices[1]].copy()
    Xdelta = population[sorted_indices[2]].copy()
    
    # El GBest es siempre Xalfa
    best_solution = Xalfa.copy()
    best_fitness = np.inf
    
    history = {
        'fitness_curve': np.zeros(max_iter),
        'diversity_curve': np.zeros(max_iter),
        'best_solution_per_iter': []
    }

    # --- 2. Evaluación Inicial (Iter 0) ---
    for i in range(pop_size):
        fitness[i] = func_objetivo(population[i])
        
    # Ordenar y actualizar Alfa, Beta, Delta después de la primera evaluación
    sorted_indices = np.argsort(fitness) # Orden MINIMIZACIÓN
    Xalfa = population[sorted_indices[0]].copy()
    Xbeta = population[sorted_indices[1]].copy()
    Xdelta = population[sorted_indices[2]].copy()
    
    best_solution = Xalfa.copy()
    best_fitness = fitness[sorted_indices[0]]

    # --- 3. Bucle Principal de Iteraciones ---
    for t in range(max_iter):
        
        # a. Calcular 'a' (parámetro de GWO)
        a = 2 - t * ((2) / max_iter)  # 'a' decrece linealmente de 2 a 0

        # b. Bucle sobre cada lobo (agente)
        for i in range(pop_size):
            
            # Generar r1 y r2 para Alfa, Beta, Delta
            r1 = np.random.rand(dim, 3) # 3 columnas (alfa, beta, delta)
            r2 = np.random.rand(dim, 3)

            # Calcular A y C
            A = 2 * a * r1 - a
            C = 2 * r2

            # Calcular distancias D
            d_alfa = np.abs(C[:, 0] * Xalfa - population[i])
            d_beta = np.abs(C[:, 1] * Xbeta - population[i])
            d_delta = np.abs(C[:, 2] * Xdelta - population[i])

            # Calcular posiciones X1, X2, X3
            X1 = Xalfa - A[:, 0] * d_alfa
            X2 = Xbeta - A[:, 1] * d_beta
            X3 = Xdelta - A[:, 2] * d_delta

            # Calcular la nueva posición (promedio)
            new_position = (X1 + X2 + X3) / 3
            
            # c. Aplicar límites de búsqueda (Clipping)
            new_position = np.clip(new_position, lb, ub)
            
            # d. Evaluar la nueva posición
            new_fitness = func_objetivo(new_position)
            
            # e. Selección Greedy (reemplazar si es mejor)
            if new_fitness < fitness[i]:
                fitness[i] = new_fitness
                population[i] = new_position.copy()

        # f. Actualizar Alfa, Beta, Delta y GBest al final de la iteración
        sorted_indices = np.argsort(fitness)
        Xalfa = population[sorted_indices[0]].copy()
        Xbeta = population[sorted_indices[1]].copy()
        Xdelta = population[sorted_indices[2]].copy()
        
        current_best_fitness = fitness[sorted_indices[0]]
        
        # Actualizar GBest si la nueva Alfa es mejor
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = Xalfa.copy()
                            
        # g. Guardar Métricas de la Iteración
        history['fitness_curve'][t] = best_fitness
        history['best_solution_per_iter'].append(best_solution.copy())
        history['diversity_curve'][t] = np.mean(np.std(population, axis=0))

        if (t + 1) % 10 == 0:
             print(f"Iteración {t + 1}/{max_iter} | Mejor Fitness: {best_fitness:.4f} | Diversidad: {history['diversity_curve'][t]:.4f}")

    # --- 4. Retornar Resultados ---
    return best_solution, best_fitness, history