import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
import sys

# Importar funções do mainActivity (certifica-te que o mainActivity.py está na mesma pasta)
from mainActivity import carregar_pessoa, extract_features_all, cal_modulos

def load_and_filter_data(root_path):
    """
    Carrega dados, extrai features, LIMPA ERROS (NaN/Inf) e filtra atividades > 7.
    """
    todos_dados = []
    participant_map = [] 

    # Percorre pastas (part1, part2, etc.)
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.endswith('.csv'):
                try:
                    # Tenta extrair ID do participante
                    part_id = int(''.join(filter(str.isdigit, f.split('dev')[0])))
                except:
                    part_id = 0 
                
                caminho = os.path.join(root, f)
                dados = carregar_pessoa(caminho)
                
                todos_dados.append(dados)
                participant_map.extend([part_id] * len(dados))

    if not todos_dados:
        print("Erro: Nenhum dado encontrado. Verifique o caminho.")
        return pd.DataFrame()

    dados_concatenados = np.vstack(todos_dados)
    participant_ids = np.array(participant_map)
    
    print("Extraindo features (Part A)... Aguarde.")
    features_df = extract_features_all(dados_concatenados[:, :9], 
                                     dados_concatenados[:, 11].astype(int), 
                                     fs=50)
    
    # --- PASSO DE LIMPEZA (CORREÇÃO DO ERRO) ---
    print("Limpando valores NaN e Infinitos...")
    # 1. Substituir Infinitos por NaN
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Preencher NaNs com a média da coluna
    features_df.fillna(features_df.mean(), inplace=True)
    
    # 3. Se ainda houver NaNs (ex: coluna inteira vazia), preencher com 0
    features_df.fillna(0, inplace=True)
    # -------------------------------------------

    # Mapear IDs para as janelas extraídas
    win_size = int(5 * 50)
    step = int(win_size * 0.5)
    n_windows = len(features_df)
    
    ids_janelados = []
    curr_idx = 0
    for _ in range(n_windows):
        if curr_idx < len(participant_ids):
            ids_janelados.append(participant_ids[curr_idx])
        else:
            ids_janelados.append(participant_ids[-1])
        curr_idx += step
    
    features_df['participant_id'] = ids_janelados

    # Filtrar atividades 1 a 7
    print("Filtrando atividades 1 a 7...")
    features_filtered = features_df[features_df['atividade'] <= 7].copy()
    
    return features_filtered

def analyze_balance(df):
    """
    1.1 Analisar o balanço entre atividades.
    """
    counts = df['atividade'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.index.astype(str), counts.values, color='skyblue')
    plt.xlabel('Atividade (Label)')
    plt.ylabel('Número de Segmentos')
    plt.title('Distribuição das Classes (Atividades 1-7)')
    plt.bar_label(bars)
    plt.show()
    
    print("\nContagem por atividade:")
    print(counts)
    
    # Resposta à pergunta 1.1
    min_c = counts.min()
    max_c = counts.max()
    ratio = max_c / min_c
    print(f"\nO dataset está balanceado? (Ratio Max/Min: {ratio:.2f})")
    if ratio > 1.5:
        print("Resposta: Não, existe desequilíbrio considerável entre classes.")
    else:
        print("Resposta: Sim, razoavelmente balanceado.")


def smote_augmentation(dataset, activity_target, k_new_samples, k_neighbors=5):
    """
    1.2 Função que implementa SMOTE para gerar K novas amostras de uma atividade A.
    
    Args:
        dataset (pd.DataFrame): Dataset completo com features e coluna 'atividade'.
        activity_target (int): A atividade A para a qual gerar dados.
        k_new_samples (int): Número de amostras sintéticas a criar.
        k_neighbors (int): Número de vizinhos a consultar (padrão 5).
        
    Returns:
        np.array: Array com as novas amostras geradas (apenas as features).
    """
    # Filtrar apenas dados da classe alvo
    class_data = dataset[dataset['atividade'] == activity_target].drop(columns=['atividade', 'participant_id'], errors='ignore')
    X = class_data.values
    
    n_samples = len(X)
    if n_samples < 2:
        raise ValueError(f"Amostras insuficientes da atividade {activity_target} para aplicar SMOTE.")
    
    # Ajustar k_neighbors se houver poucos dados
    k_neighbors = min(k_neighbors, n_samples - 1)
    
    # Modelo kNN para encontrar vizinhos
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X) # +1 porque o próprio ponto conta
    
    synthetic_samples = []
    
    for _ in range(k_new_samples):
        # 1. Escolher aleatoriamente um ponto base (índice random)
        idx_base = np.random.randint(0, n_samples)
        sample_base = X[idx_base]
        
        # 2. Encontrar vizinhos desse ponto
        # indices[0] é o próprio ponto, indices[1:] são os vizinhos
        distances, indices = nbrs.kneighbors(sample_base.reshape(1, -1))
        neighbor_indices = indices[0][1:] 
        
        # 3. Escolher aleatoriamente um dos vizinhos
        idx_neighbor = np.random.choice(neighbor_indices)
        sample_neighbor = X[idx_neighbor] # Atenção: indices do kneighbors referem-se ao X fitado? Não necessariamente se X for subset.
        # Correção: O NearestNeighbors retorna indices relativos ao array X passado no fit. 
        # Neste caso, X é class_data.values, então os índices estão corretos para X.
        sample_neighbor = X[indices[0][np.random.randint(1, len(indices[0]))]]

        # 4. Interpolação Linear
        gap = sample_neighbor - sample_base
        new_sample = sample_base + (np.random.rand() * gap)
        
        synthetic_samples.append(new_sample)
        
    return np.array(synthetic_samples)

def visualize_augmentation(df):
    """
    1.3 Gerar e visualizar 3 novas amostras da atividade 4 do participante 3[cite: 22].
    Usar apenas as duas primeiras features para o scatter plot 2D[cite: 23].
    """
    # Filtrar Participante 3
    # Nota: Certifica-te que nos teus dados tens o part3. Se não tiveres, muda o ID aqui para testar.
    target_part = 3
    target_act = 4
    n_new = 3
    
    df_part3 = df[df['participant_id'] == target_part].copy()
    
    if df_part3.empty:
        print(f"AVISO: Sem dados para o participante {target_part}. Verifica o carregamento.")
        return

    print(f"Gerando {n_new} amostras sintéticas para Part. {target_part}, Ativ. {target_act}...")

    # Gerar amostras sintéticas usando APENAS dados deste participante (como pedido)
    try:
        synthetic_feats = smote_augmentation(df_part3, target_act, n_new)
    except ValueError as e:
        print(e)
        return

    # Preparação para o Plot
    # Pegar nas 2 primeiras features (colunas 0 e 1, excluindo 'atividade' e 'id')
    feature_cols = [c for c in df_part3.columns if c not in ['atividade', 'participant_id']]
    feat1_name = feature_cols[0]
    feat2_name = feature_cols[1]
    
    plt.figure(figsize=(10, 7))
    
    # Plotar pontos originais do participante 3
    activities = df_part3['atividade'].unique()
    colors = plt.cm.get_cmap('tab10', len(activities))
    
    for i, act in enumerate(activities):
        subset = df_part3[df_part3['atividade'] == act]
        plt.scatter(subset[feat1_name], subset[feat2_name], 
                    label=f'Real - Ativ {act}', alpha=0.6, s=30)
        
    # Plotar pontos sintéticos (destaque)
    plt.scatter(synthetic_feats[:, 0], synthetic_feats[:, 1], 
                color='black', marker='X', s=150, label='Sintético - Ativ 4')
    
    plt.title(f'Data Augmentation (SMOTE) - Participante {target_part}')
    plt.xlabel(feat1_name)
    plt.ylabel(feat2_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- BLOCO PRINCIPAL PARA EXECUTAR ESTA PARTE ---
if __name__ == "__main__":
    # Caminho para os dados (Ajusta conforme o teu PC)
    path = "C:\\Users\\akcel\\OneDrive\\Ambiente de Trabalho\\school\\FCTUC\\UC_25-26\\Engenharia de Características para Aprendizagem Computacional\\Prática\\TP1"
    
    # 1. Carregar e Filtrar
    print("--- 1. Carregamento e Filtros ---")
    df_features = load_and_filter_data(path)
    
    # 2. Analisar Balanceamento
    print("\n--- 1.1 Análise de Balanceamento ---")
    analyze_balance(df_features)
    
    # 3. Visualizar Data Augmentation
    print("\n--- 1.3 Visualização SMOTE ---")
    visualize_augmentation(df_features)