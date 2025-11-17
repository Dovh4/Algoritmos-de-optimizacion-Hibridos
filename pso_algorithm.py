# Guardar como: pso_algorithm.py
import numpy as np
import random

def run_PSO(func_objetivo, dim, pop_size, max_iter, lb, ub, init_lb=None, init_ub=None):
    """
    Ejecuta el Particle Swarm Optimization (PSO) completo y autocontenido.
    
    Sigue la misma firma y estructura de run_DOA para ser intercambiable
    en los scripts de experimentación.
    
    Parámetros:
    - lb, ub: Límites de BÚSQUEDA (para clipping) (ej. -10 a 10)
    - init_lb, init_ub: Límites de INICIALIZACIÓN (opcional) (ej. -4.6 a -2.9)
    """
    
    # --- Parámetros Clásicos de PSO ---
    wMax = 0.9
    wMin = 0.4
    c1 = 2.0
    c2 = 2.0
    
    # Vmax: Límite de velocidad (heurística común: 10-20% del rango de búsqueda)
    Vmax = (ub - lb) * 0.1
    
    # --- 0. Manejo de Límites ---
    if init_lb is None:
        init_lb = lb
    if init_ub is None:
        init_ub = ub

    # --- 1. Inicialización de la población ---
    population = np.zeros((pop_size, dim))
    for i in range(pop_size):
        population[i] = init_lb + np.random.rand(dim) * (init_ub - init_lb)
    
    # Inicialización específica de PSO
    velocities = np.zeros((pop_size, dim))
    pBest_positions = population.copy()
    pBest_fitness = np.full(pop_size, np.inf)
    
    fitness = np.full(pop_size, np.inf)
    best_solution = np.zeros(dim)
    best_fitness = np.inf
    
    history = {
        'fitness_curve': np.zeros(max_iter),
        'diversity_curve': np.zeros(max_iter),
        'best_solution_per_iter': []
    }

    # --- 2. Evaluación Inicial (Iter 0) ---
    for i in range(pop_size):
        fitness[i] = func_objetivo(population[i])
        pBest_fitness[i] = fitness[i] # pBest inicial es la fitness actual
        
        if fitness[i] < best_fitness:
            best_fitness = fitness[i]
            best_solution = population[i].copy()

    # --- 3. Bucle Principal de Iteraciones ---
    for t in range(max_iter):
        
        # a. Calcular inercia (w)
        w = wMax - t * ((wMax - wMin) / max_iter)
        
        # b. Generar r1 y r2
        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)
        
        # c. Actualizar Velocidades (vectorizado)
        velocities = (
            w * velocities +
            c1 * r1 * (pBest_positions - population) +
            c2 * r2 * (best_solution - population)
        )
        
        # d. Aplicar límite de velocidad (Vmax)
        velocities = np.clip(velocities, -Vmax, Vmax)
        
        # e. Actualizar Posiciones
        population = population + velocities
        
        # f. Aplicar límites de búsqueda (lb, ub)
        population = np.clip(population, lb, ub)

        # g. Evaluar y Actualizar pBest y gBest
        for i in range(pop_size):
            fitness[i] = func_objetivo(population[i])
            
            # Actualizar pBest
            if fitness[i] < pBest_fitness[i]:
                pBest_fitness[i] = fitness[i]
                pBest_positions[i] = population[i].copy()
                
                # Actualizar gBest
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_solution = population[i].copy()
                            
        # h. Guardar Métricas de la Iteración
        history['fitness_curve'][t] = best_fitness
        history['best_solution_per_iter'].append(best_solution.copy())
        history['diversity_curve'][t] = np.mean(np.std(population, axis=0))

        if (t + 1) % 10 == 0:
             print(f"Iteración {t + 1}/{max_iter} | Mejor Fitness: {best_fitness:.4f} | Diversidad: {history['diversity_curve'][t]:.4f}")

    # --- 4. Retornar Resultados ---
    return best_solution, best_fitness, history