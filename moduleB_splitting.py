import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

# Importar ReliefF do ficheiro da Parte A
try:
    from mainActivity import reliefF
except ImportError:
    print("Aviso: mainActivity.py não encontrado. O ReliefF não vai funcionar.")

class DataManager:
    def __init__(self, df, target_col='atividade', participant_col='participant_id'):
        self.df = df
        self.target_col = target_col
        self.part_col = participant_col
        
        # Separar X (features) de y (labels) e meta-dados
        # Removemos target e participant_id das features
        self.feature_cols = [c for c in df.columns if c not in [target_col, participant_col]]
        
    def get_data_arrays(self, dataset_subset):
        """Helper para converter DataFrame em arrays X e y"""
        X = dataset_subset[self.feature_cols].values
        y = dataset_subset[self.target_col].values
        return X, y

    # --- 3.1 Divisão Within-Subject ---
    def split_within_subject(self):
        """
        Mistura todos os dados e divide: 60% Treino, 20% Val, 20% Teste.
        Usa stratify para manter a proporção de atividades.
        """
        # Primeiro separa 20% para Teste
        train_val_df, test_df = train_test_split(
            self.df, test_size=0.2, stratify=self.df[self.target_col], random_state=42
        )
        
        # Dos 80% restantes, separa 25% para Validação (25% de 80% = 20% do total)
        # Sobra 60% para Treino
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.25, stratify=train_val_df[self.target_col], random_state=42
        )
        
        return self._prepare_sets(train_df, val_df, test_df, "Within-Subject")

    # --- 3.2 Divisão Between-Subjects ---
    def split_between_subjects(self):
        """
        Divide por participantes: 9 Treino, 3 Val, 3 Teste (Total 15).
        """
        participants = self.df[self.part_col].unique()
        # Baralhar participantes
        # np.random.seed(42)
        np.random.shuffle(participants)
        
        # Tentar cumprir a regra 9-3-3. 
        # Se houver menos de 15 participantes, ajustamos proporcionalmente.
        n_p = len(participants)
        if n_p >= 15:
            train_p = participants[:9]
            val_p = participants[9:12]
            test_p = participants[12:15] # Limita a 3 se houver mais
        else:
            # Fallback proporcional se tivermos menos dados (ex: apenas 4 participantes no teste)
            n_val = max(1, int(n_p * 0.2))
            n_test = max(1, int(n_p * 0.2))
            n_train = n_p - n_val - n_test
            
            train_p = participants[:n_train]
            val_p = participants[n_train : n_train+n_val]
            test_p = participants[n_train+n_val:]
            
        print(f"  Part. Treino: {train_p}")
        print(f"  Part. Val: {val_p}")
        print(f"  Part. Teste: {test_p}")

        train_df = self.df[self.df[self.part_col].isin(train_p)]
        val_df = self.df[self.df[self.part_col].isin(val_p)]
        test_df = self.df[self.df[self.part_col].isin(test_p)]
        
        return self._prepare_sets(train_df, val_df, test_df, "Between-Subjects")

    def _prepare_sets(self, train_df, val_df, test_df, strategy_name):
        """
        Organiza os DataFrames em dicionários X, y para facilitar o processamento.
        """
        X_train, y_train = self.get_data_arrays(train_df)
        X_val, y_val = self.get_data_arrays(val_df)
        X_test, y_test = self.get_data_arrays(test_df)
        
        print(f"\n--- Estratégia: {strategy_name} ---")
        print(f"Train shape: {X_train.shape}")
        print(f"Val shape:   {X_val.shape}")
        print(f"Test shape:  {X_test.shape}")
        
        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)
        }

    # --- 3.4 Processamento (Cenários a, b, c) ---
    def process_scenarios(self, split_data):
        """
        Aplica Normalização, PCA e Feature Selection.
        IMPORTANTE: Fit apenas no Treino, Transform no resto.
        """
        X_train, y_train = split_data["train"]
        X_val, y_val = split_data["val"]
        X_test, y_test = split_data["test"]
        
        scenarios = {}
        
        # 0. Normalização (Sempre necessária antes de PCA/KNN)
        scaler = StandardScaler()
        # FIT apenas no treino!
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)
        X_test_norm = scaler.transform(X_test)
        
        # --- Cenário A: Todas as features (Normalizadas) ---
        scenarios["All Features"] = (X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test)
        
        # --- Cenário B: PCA (90% Variância) ---
        # FIT apenas no treino!
        pca = PCA(n_components=0.90)
        X_train_pca = pca.fit_transform(X_train_norm)
        X_val_pca = pca.transform(X_val_norm)
        X_test_pca = pca.transform(X_test_norm)
        
        scenarios[f"PCA ({pca.n_components_} comps)"] = (X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test)
        
        # --- Cenário C: ReliefF (Top 15) ---
        # Otimização: Usar sample se for muito grande
        if len(X_train_norm) > 2000:
            idx = np.random.choice(len(X_train_norm), 2000, replace=False)
            X_sample = X_train_norm[idx]
            y_sample = y_train[idx]
        else:
            X_sample = X_train_norm
            y_sample = y_train
            
        print("  Calculando ReliefF (Top 15)...")
        # Chama a função importada do mainActivity
        scores = reliefF(X_sample, y_sample, n_neighbors=10)
        
        # Selecionar índices das top 15
        top_indices = np.argsort(scores)[::-1][:15]
        
        X_train_rel = X_train_norm[:, top_indices]
        X_val_rel = X_val_norm[:, top_indices]
        X_test_rel = X_test_norm[:, top_indices]
        
        scenarios["ReliefF (Top 15)"] = (X_train_rel, y_train, X_val_rel, y_val, X_test_rel, y_test)
        
        return scenarios

# Exemplo de utilização (Podes colar no final para testar)
if __name__ == "__main__":
    # Simulação: Carregar os datasets que geraste antes
    # Assumindo que já tens df_features e df_emb na memória ou gravados
    pass