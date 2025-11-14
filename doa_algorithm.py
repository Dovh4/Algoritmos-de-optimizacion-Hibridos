# Guardar como: doa_algorithm.py
import numpy as np
import random

def run_DOA(func_objetivo, dim, pop_size, max_iter, lb, ub, init_lb=None, init_ub=None):
    """
    Ejecuta el Dhole Optimization Algorithm (DOA) completo y autocontenido.
    
    Parámetros:
    - lb, ub: Límites de BÚSQUEDA (para clipping) (ej. -10 a 10)
    - init_lb, init_ub: Límites de INICIALIZACIÓN (opcional) (ej. -4.6 a -2.9)
    """
    
    # --- 0. Manejo de Límites ---
    
    # Si no se dan límites de inicialización, usar los de búsqueda
    if init_lb is None:
        init_lb = lb
    if init_ub is None:
        init_ub = ub
    
    # 1. Inicialización de la población (CON LÍMITES 'init_')
    population = np.zeros((pop_size, dim))
    for i in range(pop_size):
        population[i] = init_lb + np.random.rand(dim) * (init_ub - init_lb)
        
    fitness = np.full(pop_size, np.inf)
    best_solution = np.zeros(dim)
    best_fitness = np.inf
    
    history = {
        'fitness_curve': np.zeros(max_iter),
        'diversity_curve': np.zeros(max_iter),
        'best_solution_per_iter': []
    }

    # 2. Evaluación Inicial (Iter 0)
    for i in range(pop_size):
        fitness[i] = func_objetivo(population[i])
        
        if fitness[i] < best_fitness:
            best_fitness = fitness[i]
            best_solution = population[i].copy()

    # 3. Bucle Principal de Iteraciones
    for t in range(max_iter):
        
        # Genera una población candidata
        poblacion_candidata = _iterarDHOA(
            maxIter=max_iter,
            iter=t,
            dim=dim,
            population=population,
            fitness=fitness,
            best=best_solution,
            func_objetivo=func_objetivo, 
            lb=lb, # <-- Pasa los límites de BÚSQUEDA (ej. -10)
            ub=ub  # <-- Pasa los límites de BÚSQUEDA (ej. 10)
        )

        # 3b. Evaluar y Seleccionar (Selección Greedy)
        for i in range(pop_size):
            nuevo_fitness = func_objetivo(poblacion_candidata[i])
            
            if nuevo_fitness < fitness[i]:
                fitness[i] = nuevo_fitness
                population[i] = poblacion_candidata[i].copy()
                
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_solution = population[i].copy()
                            
        # 3c. Guardar Métricas de la Iteración
        history['fitness_curve'][t] = best_fitness
        history['best_solution_per_iter'].append(best_solution.copy())
        history['diversity_curve'][t] = np.mean(np.std(population, axis=0))

        if (t + 1) % 10 == 0:
             print(f"Iteración {t + 1}/{max_iter} | Mejor Fitness: {best_fitness:.4f} | Diversidad: {history['diversity_curve'][t]:.4f}")

    # 4. Retornar Resultados
    return best_solution, best_fitness, history


def _iterarDHOA(
    maxIter,
    iter,
    dim,
    population,
    fitness,
    best,
    func_objetivo,
    lb,
    ub
):
    """
    Función interna: Genera una población candidata usando UNA iteración del DOA.
    """

    # ... (Cálculos de pmn, presa_objetivo, c2, ps, etc. sin cambios) ...
    num_dholes = population.shape[0]
    poblacion_candidata = np.zeros((num_dholes, dim)) 
    mejor_indice_local = np.argmin(fitness)
    prey_local = population[mejor_indice_local]
    pmn = round(random.random() * 15 + 5)
    presa_objetivo = (prey_local + best) / 2.0
    c2 = 1 - (iter / maxIter)
    denominador_ps = 1 + np.exp(-0.5 * (pmn - 25))
    ps = (1.0 / denominador_ps) * 1.0
    fitness_presa_objetivo = func_objetivo(presa_objetivo) 
    epsilon = 1e-20
    
    # 2. Bucle principal sobre cada Dhole (i)
    for i in range(num_dholes):
        posicion_actual = population[i]
        fitness_actual = fitness[i]
        nueva_posicion_i = np.zeros(dim) 
        vocalizacion = random.random()
        z = i
        while z == i:
            z = random.randint(0, num_dholes - 1)
        posicion_z = population[z]
        rand_s = random.random()
        if fitness_presa_objetivo < epsilon:
            tamano_S = 0.0 if fitness_actual < epsilon else float('inf')
        else:
            tamano_S = 3 * rand_s * (fitness_actual / fitness_presa_objetivo)
        presa_debilitada = None
        if tamano_S > 2:
            if tamano_S == float('inf'):
                factor_debilitamiento = 1.0
            else:
                factor_debilitamiento = np.exp(-1.0 / tamano_S)
            presa_debilitada = factor_debilitamiento * prey_local

        
        # 3. Bucle por Dimensión (j) - Ecuaciones de Movimiento
        for j in range(dim):
            
            if vocalizacion < 0.5: # (Mantenemos tu cambio de 70% exploración)
                # Fase de Exploración
                if pmn < 10:
                    rand_busqueda = random.random()
                    movimiento_j = c2 * rand_busqueda * (presa_objetivo[j] - posicion_actual[j])
                    nueva_posicion_i[j] = posicion_actual[j] + movimiento_j
                else:
                    nueva_posicion_i[j] = posicion_actual[j] - posicion_z[j] + presa_objetivo[j]
            else:
                # Fase de Explotación
                if tamano_S > 2:
                    rand_ataque_grande = random.random()
                    cos_term = np.cos(2 * np.pi * rand_ataque_grande)
                    sin_term = np.sin(2 * np.pi * rand_ataque_grande)
                    W_prey_j = presa_debilitada[j]
                    movimiento_j = W_prey_j * ps * (cos_term - sin_term * W_prey_j * ps)
                    nueva_posicion_i[j] = posicion_actual[j] + movimiento_j
                else:
                    rand_ataque_peq = random.random()
                    termino1_j = (posicion_actual[j] - best[j]) * ps
                    termino2_j = ps * rand_ataque_peq * posicion_actual[j]
                    nueva_posicion_i[j] = termino1_j + termino2_j
                    
        # 4. Aplicar límites (clipping)
        # USA LOS LÍMITES DE BÚSQUEDA (lb, ub) (ej. -10 y 10)
        nueva_posicion_i = np.clip(nueva_posicion_i, lb, ub)
        poblacion_candidata[i] = nueva_posicion_i

    return poblacion_candidata