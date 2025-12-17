import numpy as np
import pandas as pd
import torch
import os
import sys

# Importar as ferramentas fornecidas
# Certifica-te que embeddings_extractor.py está na mesma pasta
try:
    from embeddings_extractor import load_model, resample_to_30hz_5s
    from mainActivity import carregar_pessoa
except ImportError as e:
    print("ERRO: Certifique-se que 'embeddings_extractor.py' e 'mainActivity.py' estão na pasta.")
    print(e)
    sys.exit(1)

def acc_segmentation_adapted(data, fs_input=50):
    '''
    Adaptação da função acc_segmentation para garantir flexibilidade no Sampling Rate.
    Segmenta os dados em janelas de 5 segundos.
    '''
    TIMESTAMP_COL = 10
    LABEL_COL = 11
    MIN_SEGMENT_SIZE = 20 # Mínimo de pontos para considerar válido
    
    # Tamanho da janela em timestamps (segundos * fs não se aplica diretamente ao timestamp se este for tempo absoluto)
    # Mas o código original usava 'win_size = 5000' numa coluna de timestamp. 
    # Assumindo que o timestamp é numérico contínuo ou milissegundos.
    # Vamos usar a lógica de tempo: janela de 5 segundos.
    
    start_time = data[0, TIMESTAMP_COL]
    # Se o timestamp estiver em segundos ou ms, precisamos ajustar. 
    # Normalmente, nestes datasets, o intervalo depende da unidade. 
    # O código original usava win_size=5000 (provavelmente ms? 5s = 5000ms).
    win_size_time = 5000 # Assumindo unidade compatível com o código original (provavelmente ms)
    
    end_time = start_time + win_size_time
    
    activities = []
    segments = []
    
    # Loop de segmentação
    while end_time < data[-1, TIMESTAMP_COL]:
        mask = (data[:, TIMESTAMP_COL] >= start_time) & (data[:, TIMESTAMP_COL] < end_time)
        
        # Verifica se temos dados suficientes e se a atividade é constante na janela
        if np.sum(mask) > MIN_SEGMENT_SIZE:
            labels_in_window = data[mask, LABEL_COL]
            
            # Verifica se a atividade é única na janela
            if np.all(labels_in_window == labels_in_window[0]):
                activity = labels_in_window[0]
                
                # Filtrar AQUI as atividades > 7 para poupar processamento
                if activity <= 7:
                    # Extrair Acelerómetro (colunas 1, 2, 3 no CSV lido pelo carregar_pessoa -> indices 1,2,3)
                    acc_xyz = data[mask, 1:4]
                    
                    activities.append(activity)
                    segments.append(acc_xyz)
        
        # Janelas sem sobreposição (conforme lógica original: start = end - win/2 para 50% overlap?)
        # O código original fazia: start_time = end_time - win_size/2 (Overlap de 50%)
        start_time = end_time - (win_size_time / 2) 
        end_time = start_time + win_size_time
        
    return segments, activities

def create_embeddings_dataset(root_path):
    print("--- A carregar modelo Harnet5 (pode demorar na 1ª vez) ---")
    feature_encoder = load_model()
    
    all_embeddings = []
    all_activities = []
    all_participants = []
    
    print(f"--- A processar ficheiros em: {root_path} ---")
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.endswith('.csv'):
                try:
                    part_id = int(''.join(filter(str.isdigit, f.split('dev')[0])))
                except:
                    part_id = 0
                
                caminho = os.path.join(root, f)
                # Carregar dados brutos
                dados = carregar_pessoa(caminho)
                
                # 1. Segmentar (Apenas Acelerómetro e Atividades <= 7)
                # Assumimos fs=50Hz conforme Part A (ajusta se o dataset for diferente)
                segments, activities = acc_segmentation_adapted(dados, fs_input=50)
                
                if not segments:
                    continue
                
                # 2. Reamostrar para 30Hz
                # O modelo espera 30Hz. A função resample recebe (N, 3) e devolve (M, 3) a 30Hz
                resampled_segments = []
                valid_indices = []
                
                for i, seg in enumerate(segments):
                    # fs_in_hz=50 (dataset original). O output será a 30Hz.
                    res_seg, _ = resample_to_30hz_5s(seg, fs_in_hz=50.0)
                    
                    # Verificação de segurança: O modelo espera tamanho fixo (30Hz * 5s = 150 samples)
                    if res_seg.shape[0] == 150:
                        resampled_segments.append(res_seg)
                        valid_indices.append(i)
                
                if not resampled_segments:
                    continue
                
                # Atualizar listas de metadados apenas para os segmentos válidos
                activities = [activities[i] for i in valid_indices]
                current_part_ids = [part_id] * len(activities)
                
                # 3. Gerar Embeddings em Batch
                # Converter para numpy array: (Batch, Time, Channels)
                x_batch = np.array(resampled_segments)
                
                # O modelo espera (Batch, Channels, Time) -> Transpor eixos 1 e 2
                x_batch = np.transpose(x_batch, (0, 2, 1)) # shape: (B, 3, 150)
                
                # Passar pelo modelo
                with torch.no_grad():
                    # Converter para Tensor Float
                    t_input = torch.from_numpy(x_batch).float()
                    
                    # Extrair embeddings
                    # O output do feature_extractor é (Batch, 512) ou similar
                    embeddings = feature_encoder(t_input)
                    
                    # Guardar
                    all_embeddings.append(embeddings.cpu().numpy())
                    all_activities.extend(activities)
                    all_participants.extend(current_part_ids)
                    
                print(f"Processado: {f} -> {len(activities)} segmentos.")

    # Consolidar tudo
    if not all_embeddings:
        print("Nenhum segmento válido encontrado.")
        return None, None
        
    X_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # --- CORREÇÃO: Remover dimensão extra (squeeze) ---
    # O shape atual é (N, 512, 1), queremos (N, 512)
    if X_embeddings.ndim == 3:
        X_embeddings = np.squeeze(X_embeddings)
    # --------------------------------------------------

    # Criar DataFrame final
    # As colunas serão emb_0, emb_1, ... emb_N
    cols = [f'emb_{i}' for i in range(X_embeddings.shape[1])]
    df_embeddings = pd.DataFrame(X_embeddings, columns=cols)
    
    # Adicionar Labels e Participantes (Crucial para a Parte 3)
    df_embeddings['atividade'] = all_activities
    df_embeddings['participant_id'] = all_participants

    # --- NOVO: APRESENTAR RESULTADOS EM TABELA ---
    print("\n" + "="*60)
    print("          RESULTADO: EMBEDDINGS DATASET")
    print("="*60)
    
    # Dimensões exatas excluindo as colunas de metadados para mostrar [n_seg, n_emb]
    n_seg = df_embeddings.shape[0]
    n_emb = df_embeddings.shape[1] - 2 # subtrair 'atividade' e 'participant_id'
    
    print(f"Dimensões Finais: [{n_seg}, {n_emb}] (n_segmentos, n_embeddings)")
    print("-" * 60)
    print("Amostra dos Dados (Primeiras 5 linhas):")
    
    # Configurar pandas para mostrar a tabela bonita no terminal
    pd.set_option('display.max_columns', 6) # Mostra apenas algumas colunas
    pd.set_option('display.width', 1000)
    print(df_embeddings.head())
    print("="*60 + "\n")
    # ---------------------------------------------
    
    return df_embeddings

if __name__ == "__main__":
    # Ajusta o caminho para a tua pasta de dados
    path = "C:\\Users\\akcel\\OneDrive\\Ambiente de Trabalho\\school\\FCTUC\\UC_25-26\\Engenharia de Características para Aprendizagem Computacional\\Prática\\TP1"
    
    print("--- Início Módulo B: Secção 2 (Embeddings) ---")
    df_emb = create_embeddings_dataset(path)
    
    if df_emb is not None:
        print("\nSucesso!")
        print(f"Shape do Dataset de Embeddings: {df_emb.shape}")
        print(df_emb.head())
        
        # Opcional: Guardar em CSV para não ter de recalcular sempre
        # df_emb.to_csv("embeddings_dataset.csv", index=False)
        # print("Dataset guardado em 'embeddings_dataset.csv'")