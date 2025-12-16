import numpy as np
import pandas as pd
# Importar as funções de extração que já fizeste na Parte A
from mainActivity import extract_temporal_features, extract_spectral_features, extract_physical_features

def extract_features_single_segment(segment, fs=50):
    """
    Extrai as features de UM segmento único (ex: 256x9).
    Deve seguir EXATAMENTE a mesma ordem de colunas do treino (Part A).
    """
    # Lista para guardar os valores ordenados
    feat_values = []
    feat_names = [] # Só para debug, se precisares
    
    # 1. Features por Eixo (Acc, Gyr, Mag)
    # Ordem das colunas no raw: 0,1,2=Acc | 3,4,5=Gyr | 6,7,8=Mag
    sensores = ["acc", "gyr", "mag"]
    
    for i, sensor in enumerate(sensores):
        base_idx = i * 3
        for eixo, label in zip(range(3), ['x', 'y', 'z']):
            col_idx = base_idx + eixo
            sinal = segment[:, col_idx]

            # Temporais
            t_feats = extract_temporal_features(sinal)
            # A ordem de inserção aqui é crucial: tem de ser igual ao dicionário
            # Como dicionários modernos mantêm ordem de inserção, iteramos as keys
            for k, v in t_feats.items():
                feat_values.append(v)
                feat_names.append(f"{sensor}_{label}_{k}")

            # Espectrais
            f_feats = extract_spectral_features(sinal, fs)
            for k, v in f_feats.items():
                feat_values.append(v)
                feat_names.append(f"{sensor}_{label}_{k}")

    # 2. Features Físicas (Módulos e Vetores 3D)
    acc_window = segment[:, 0:3]
    gyr_window = segment[:, 3:6]
    
    phys_feats = extract_physical_features(acc_window, gyr_window, fs)
    for k, v in phys_feats.items():
        feat_values.append(v)
        feat_names.append(k)
        
    # Retorna como array 2D (1 linha, N features)
    return np.array(feat_values).reshape(1, -1)

def deploy_system(raw_segment, saved_scaler, saved_model, saved_pca=None, saved_selector=None):
    """
    Função Final de Deployment.
    
    Args:
        raw_segment: Array (N, 9) com dados crus.
        saved_scaler: O objeto StandardScaler treinado (fitted).
        saved_model: O modelo kNN treinado.
        saved_pca: (Opcional) O objeto PCA treinado.
        saved_selector: (Opcional) O objeto ReliefF/SelectKBest treinado.
        
    Returns:
        String com o nome da atividade prevista.
    """
    # 1. Validação do Input
    if raw_segment.shape[1] != 9:
        return "Erro: O segmento deve ter 9 colunas (AccXYZ, GyrXYZ, MagXYZ)."
    
    # 2. Extração de Features (A parte que faltava!)
    try:
        # Extrair vetor de features (1 x 138 aprox)
        features_vector = extract_features_single_segment(raw_segment, fs=50)
        
        # Tratar NaNs se existirem (substituir por 0 ou média global se tivesses guardado)
        features_vector = np.nan_to_num(features_vector)
        
    except Exception as e:
        return f"Erro na extração de features: {e}"

    # 3. Normalização (Obrigatório usar o scaler do treino)
    try:
        X_scaled = saved_scaler.transform(features_vector)
    except Exception as e:
        return f"Erro na normalização (verifique dimensões): {e}"

    # 4. Redução de Dimensionalidade / Seleção (Se usaste no treino)
    X_final = X_scaled
    if saved_pca is not None:
        X_final = saved_pca.transform(X_final)
    elif saved_selector is not None:
        # Se for ReliefF manual, terias de ter guardado os índices das colunas (top_indices)
        # Aqui assumimos que saved_selector é um objeto que tem .transform
        X_final = saved_selector.transform(X_final)

    # 5. Classificação
    prediction_idx = saved_model.predict(X_final)[0]
    
    # 6. Mapeamento para Texto
    # (Ajusta este dicionário se os teus labels forem diferentes, ex: 0-6 ou 1-7)
    activity_map = {
        1: "Standing (1)", 
        2: "Sitting (2)", 
        3: "Sit & Talk (3)", 
        4: "Walking (4)", 
        5: "Walk & Talk (5)", 
        6: "Climbing Stairs (6)", 
        7: "Climb & Talk (7)"
    }
    
    label = activity_map.get(prediction_idx, f"Desconhecido ({prediction_idx})")
    
    return label

# --- BLOCO DE TESTE ---
if __name__ == "__main__":
    # Vamos simular que temos os modelos na memória (vêm do moduleB_main.py)
    # Se estiveres a correr isto separado, este teste vai falhar porque precisa dos modelos.
    # Mas podes importar este ficheiro no teu 'main.py' e testar lá.
    
    print("Função deploy_system pronta a usar.")
    print("Para testar, importa esta função no teu script principal onde tens o 'workflow_f.final_model'.")