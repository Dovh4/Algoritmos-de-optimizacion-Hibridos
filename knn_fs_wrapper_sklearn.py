# Guardar como: knn_fs_wrapper.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

class KnnFitness:
    """
    Wrapper que conecta el DOA con el kNN.
    Maneja la binarización y la evaluación de fitness usando Hold-Out.
    """
    
    def __init__(self, X_train, y_train, k_vecinos=5, alpha=0.9):
        """
        Inicializa el evaluador y divide los datos de entrenamiento
        en sub-train y sub-validación (Hold-Out).
        """
        self.dim = X_train.shape[1] 
        self.k_vecinos = k_vecinos
        self.alpha = alpha
        
        # Partición Hold-Out para la evaluación interna del fitness
        self.X_train_fit, self.X_val_fit, self.y_train_fit, self.y_val_fit = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        self.knn = KNeighborsClassifier(n_neighbors=self.k_vecinos)

    def _binarizar(self, solucion_continua):
        """ Binariza la solución continua usando S-Shape y ruleta (Estocástico). """
        probabilidades = 1 / (1 + np.exp(-solucion_continua))
        solucion_binaria = (probabilidades > np.random.rand(self.dim)).astype(int)
        return solucion_binaria

    def evaluate(self, solucion_continua):
        """
        Función de fitness principal (usada por el DOA).
        Utiliza binarización estocástica para la exploración.
        """
        
        # a. Binarización estocástica
        solucion_binaria = self._binarizar(solucion_continua)
        
        # b. Manejar caso "cero características"
        num_caracteristicas = np.sum(solucion_binaria)
        if num_caracteristicas == 0:
            return 1.0 # Penalización máxima

        # c. Evaluar kNN con Hold-Out
        X_train_subset = self.X_train_fit[:, solucion_binaria == 1]
        X_val_subset = self.X_val_fit[:, solucion_binaria == 1]
        
        try:
            self.knn.fit(X_train_subset, self.y_train_fit)
            preds = self.knn.predict(X_val_subset)
            precision_media = accuracy_score(self.y_val_fit, preds)
        except ValueError:
            precision_media = 0.0 # Penaliza si el subset es inválido
        
        # d. Calcular Fitness (Minimización)
        error = 1.0 - precision_media
        reduccion = num_caracteristicas / self.dim
        fitness = (self.alpha * error) + ((1 - self.alpha) * reduccion)
        
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
            return 1.0, 0.0, 0 # error, reduccion_pct, num_feat

        # Evaluar con la misma partición Hold-Out
        X_train_subset = self.X_train_fit[:, solucion_binaria == 1]
        X_val_subset = self.X_val_fit[:, solucion_binaria == 1]
        
        try:
            self.knn.fit(X_train_subset, self.y_train_fit)
            preds = self.knn.predict(X_val_subset)
            precision_media = accuracy_score(self.y_val_fit, preds)
        except ValueError:
            precision_media = 0.0
        
        error = 1.0 - precision_media
        reduccion_pct = num_caracteristicas / self.dim
        
        return error, reduccion_pct, num_caracteristicas