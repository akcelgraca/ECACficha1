import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from scipy.stats import ttest_rel

class KNNWorkflow:
    def __init__(self, max_k=21):
        self.max_k = max_k
        self.best_k = None
        self.final_model = None
        
    def tune_and_train(self, X_train, y_train, X_val, y_val):
        """
        4.1 & 5.2: Seleciona o melhor k usando o conjunto de validação.
        """
        best_score = -1
        best_k = 1
        
        # Testar k ímpares de 1 até max_k
        possible_ks = range(1, self.max_k + 1, 2)
        
        for k in possible_ks:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
            knn.fit(X_train, y_train)
            score = knn.score(X_val, y_val)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        self.best_k = best_k
        # print(f"    -> Melhor k encontrado: {best_k} (Val Acc: {best_score:.4f})")
        
        # 5.3: Retreinar com (Train + Val) usando o melhor k
        X_combined = np.vstack((X_train, X_val))
        y_combined = np.concatenate((y_train, y_val))
        
        self.final_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
        self.final_model.fit(X_combined, y_combined)
        
        return best_k

    def evaluate(self, X_test, y_test, plot_cm=False, title=""):
        """
        4.2 & 5.3: Avalia no set de teste e devolve métricas.
        """
        if self.final_model is None:
            raise Exception("Modelo não treinado. Execute tune_and_train primeiro.")
            
        y_pred = self.final_model.predict(X_test)
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        if plot_cm:
            cm = confusion_matrix(y_test, y_pred)
            
            # --- CORREÇÃO 1: Imprimir Matriz em Texto também ---
            print("\nMatriz de Confusão (Texto):")
            print(cm)
            print("-" * 30)
            # ---------------------------------------------------

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix: {title}\n(Acc: {acc:.2f}, k={self.best_k})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # --- CORREÇÃO 2: Garantir que o plot é bloqueante ---
            plt.show(block=True) 
            
        return {
            "accuracy": acc,
            "f1_score": f1,
            "y_true": y_test,
            "y_pred": y_pred,
            "best_k": self.best_k
        }

def perform_hypothesis_test(scores_model_A, scores_model_B):
    """
    5.4: Teste estatístico (t-test emparelhado) para comparar dois modelos.
    Recebe listas de accuracies de várias execuções.
    """
    stat, p_val = ttest_rel(scores_model_A, scores_model_B)
    print(f"\n--- Teste de Hipótese (T-Test) ---")
    print(f"Média Modelo A: {np.mean(scores_model_A):.4f} | Média Modelo B: {np.mean(scores_model_B):.4f}")
    print(f"P-value: {p_val:.5e}")
    
    if p_val < 0.05:
        print(">> Diferença ESTATISTICAMENTE SIGNIFICATIVA.")
    else:
        print(">> Não há diferença estatística significativa.")