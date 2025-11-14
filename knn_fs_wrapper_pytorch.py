# Guardar como: knn_fs_wrapper_pytorch.py
import numpy as np
import torch
from sklearn.model_selection import train_test_split 

# --- Configuración del Dispositivo (GPU o CPU) ---
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("PyTorch: Usando GPU (CUDA).")
else:
    device = torch.device('cpu')
    print("PyTorch: Usando CPU.")

# --- Función kNN personalizada en PyTorch ---
def knn_pytorch(X_train, y_train, X_test, k):
    """
    Implementación de kNN usando tensores de PyTorch.
    Mueve los datos a la GPU, calcula distancias, vota y devuelve predicciones.
    """
    
    # 1. Mover datos a la GPU como tensores
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    
    # 2. Calcular distancias Euclidianas
    distancias = torch.cdist(X_test_t, X_train_t, p=2.0)
    
    # 3. Encontrar los k vecinos más cercanos (índices)
    _, indices_vecinos = torch.topk(distancias, k, dim=1, largest=False)
    
    # 4. Obtener las etiquetas de esos vecinos
    etiquetas_vecinos = torch.gather(y_train_t.expand(X_test_t.shape[0], -1), 1, indices_vecinos)
    
    # 5. Votación (Moda)
    predicciones, _ = torch.mode(etiquetas_vecinos, dim=1)
    
    return predicciones

# --- Clase Wrapper ---
class KnnFitness:
    """ 
    Wrapper de kNN para GPU (usando PyTorch). 
    """
    
    def __init__(self, X_train, y_train, k_vecinos=5, alpha=0.9):
        """
        Inicializa el evaluador y divide los datos de entrenamiento
        en sub-train y sub-validación (Hold-Out).
        """
        self.dim = X_train.shape[1] 
        self.k_vecinos = k_vecinos
        self.alpha = alpha
        
        # Partición Hold-Out (en CPU, como arrays de Numpy)
        self.X_train_fit, self.X_val_fit, self.y_train_fit, self.y_val_fit = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

    def _binarizar(self, solucion_continua):
        """ Binariza la solución continua usando S-Shape y ruleta (Estocástico). """
        probabilidades = 1 / (1 + np.exp(-solucion_continua))
        solucion_binaria = (probabilidades > np.random.rand(self.dim)).astype(int)
        return solucion_binaria

    def evaluate(self, solucion_continua):
        """
        Función de fitness principal (usada por el DOA).
        """
        
        solucion_binaria = self._binarizar(solucion_continua)
        num_caracteristicas = np.sum(solucion_binaria)
        if num_caracteristicas == 0:
            return 1.0 

        X_train_subset = self.X_train_fit[:, solucion_binaria == 1]
        X_val_subset = self.X_val_fit[:, solucion_binaria == 1]
        
        try:
            preds_tensor = knn_pytorch(
                X_train_subset, 
                self.y_train_fit, 
                X_val_subset, 
                self.k_vecinos
            )
            preds_numpy = preds_tensor.cpu().numpy()
            correctos = np.sum(preds_numpy == self.y_val_fit)
            precision_media = correctos / len(self.y_val_fit)

        except (ValueError, RuntimeError): 
            precision_media = 0.0
        
        # --- Cálculo de Fitness (Minimización) ---
        error = 1.0 - precision_media
        
        # Estrategia 1: Minimizar N° de Características (Original)
        reduccion = num_caracteristicas / self.dim
        fitness = (self.alpha * error) + ((1 - self.alpha) * reduccion)
        
        # Estrategia 2: Perseguir 20 Características
        #TARGET_FEATURES = 20
        # Penaliza la distancia (absoluta) al objetivo de 20
        #penalizacion_tamano = abs(num_caracteristicas - TARGET_FEATURES) / self.dim
        #fitness = (self.alpha * error) + ((1 - self.alpha) * penalizacion_tamano)
        
        return fitness

    def _binarizar_determinista(self, solucion_continua):
        """ Binariza la solución continua usando S-Shape y UMBRAL FIJO 0.5 """
        probabilidades = 1 / (1 + np.exp(-solucion_continua))
        solucion_binaria = (probabilidades > 0.5).astype(int) 
        return solucion_binaria

    def evaluate_components(self, solucion_binaria):
        """
        Evalúa una solución binaria (para el reporte) y devuelve 
        sus componentes (error, reduccion) por separado.
        """
        num_caracteristicas = np.sum(solucion_binaria)
        if num_caracteristicas == 0:
            return 1.0, 0.0, 0 

        X_train_subset = self.X_train_fit[:, solucion_binaria == 1]
        X_val_subset = self.X_val_fit[:, solucion_binaria == 1]
        
        try:
            preds_tensor = knn_pytorch(
                X_train_subset, 
                self.y_train_fit, 
                X_val_subset, 
                self.k_vecinos
            )
            preds_numpy = preds_tensor.cpu().numpy()
            correctos = np.sum(preds_numpy == self.y_val_fit)
            precision_media = correctos / len(self.y_val_fit)
            
        except (ValueError, RuntimeError):
            precision_media = 0.0
        
        error = 1.0 - precision_media
        
        # Estrategia 1: Minimizar N° de Características (Original)
        reduccion_pct = num_caracteristicas / self.dim
        
        # Estrategia 2: Perseguir 20 Características
        #TARGET_FEATURES = 20
        # Reporta la penalización (distancia a 20) en lugar de la reducción
        #reduccion_pct = abs(num_caracteristicas - TARGET_FEATURES) / self.dim
        
        return error, reduccion_pct, num_caracteristicas