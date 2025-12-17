"""
Akcel Soares da Graça
Yel
Edson Alage
"""

import pandas as pd
import numpy as np
import time
import os

# Importar os módulos criados anteriormente
from moduleB_augmentation import load_and_filter_data
from moduleB_embeddings import create_embeddings_dataset
from moduleB_splitting import DataManager
from moduleB_model import KNNWorkflow, perform_hypothesis_test
from moduleB_deployment import deploy_system
# Importar carregar_pessoa para ler os dados brutos reais
from mainActivity import carregar_pessoa 
from sklearn.preprocessing import StandardScaler

def run_full_evaluation(n_repeats=3):
    # --- 1. Carregamento de Dados ---
    path = "C:\\Users\\akcel\\OneDrive\\Ambiente de Trabalho\\school\\FCTUC\\UC_25-26\\Engenharia de Características para Aprendizagem Computacional\\Prática\\TP1"
    
    print("\n[1/5] A preparar DATASETS...")
    
    # Dataset A: Features Manuais
    print(" -> Carregando Features Dataset (Parte A)...")
    df_features = load_and_filter_data(path)
    
    # Dataset B: Embeddings
    print(" -> Carregando Embeddings Dataset (Parte B)...")
    df_embeddings = create_embeddings_dataset(path)
    
    if df_embeddings is None:
        print("Erro crítico: Embeddings não geradas.")
        return

    results_features = []
    results_embeddings = []

    # Variáveis para guardar o último set de teste e modelo (para o plot final e deployment)
    last_X_te_f = None
    last_y_te_f = None
    last_workflow_f = None

    print(f"\n[2/5] A iniciar LOOP de Avaliação ({n_repeats} repetições)...")
    
    for i in range(n_repeats):
        print(f"\n--- REPETIÇÃO {i+1}/{n_repeats} ---")
        
        # --- Aleatoriedade ---
        np.random.seed(int(time.time()) + i)
        
        # Gestão de Dados e Splits
        dm_feats = DataManager(df_features)
        dm_embs = DataManager(df_embeddings)
        
        splits_feats = dm_feats.split_between_subjects()
        splits_embs = dm_embs.split_between_subjects()
        
        scenarios_feats = dm_feats.process_scenarios(splits_feats)
        scenarios_embs = dm_embs.process_scenarios(splits_embs)
        
        # --- A: FEATURES ---
        print(" -> Treinando Modelo: Features (Original)...")
        workflow_f = KNNWorkflow()
        X_tr_f, y_tr_f, X_val_f, y_val_f, X_te_f, y_te_f = scenarios_feats["All Features"]
        
        workflow_f.tune_and_train(X_tr_f, y_tr_f, X_val_f, y_val_f)
        res_f = workflow_f.evaluate(X_te_f, y_te_f, plot_cm=False)
        results_features.append(res_f['accuracy'])
        print(f"    Acc: {res_f['accuracy']:.2%}")

        # Guardar referências da última iteração
        last_X_te_f = X_te_f
        last_y_te_f = y_te_f
        last_workflow_f = workflow_f

        # --- B: EMBEDDINGS ---
        print(" -> Treinando Modelo: Embeddings (Original)...")
        workflow_e = KNNWorkflow()
        X_tr_e, y_tr_e, X_val_e, y_val_e, X_te_e, y_te_e = scenarios_embs["All Features"]
        
        workflow_e.tune_and_train(X_tr_e, y_tr_e, X_val_e, y_val_e)
        res_e = workflow_e.evaluate(X_te_e, y_te_e, plot_cm=False)
        results_embeddings.append(res_e['accuracy'])
        print(f"    Acc: {res_e['accuracy']:.2%}")

    # --- 3. Análise Estatística ---
    print("\n[3/5] Resultados Finais e Teste Estatístico (Between-Subjects)")
    perform_hypothesis_test(results_features, results_embeddings)
    
    # --- 4. Plot Final ---
    print("\n[4/5] Exemplo visual: Matriz de Confusão (Features - Última Run)")
    if last_workflow_f is not None:
        last_workflow_f.evaluate(last_X_te_f, last_y_te_f, plot_cm=True, title="Features - Between Subjects")

    # --- 5. TESTE DE DEPLOYMENT COM DADOS REAIS ---
    print("\n[5/5] Demonstração de DEPLOYMENT (Dados Reais)")
    
    if last_workflow_f is not None:
        model_final = last_workflow_f.final_model
        
        # 1. Recriar o Scaler usando TODOS os dados disponíveis
        # (Em produção, carregarias um ficheiro 'scaler.pkl', aqui treinamos na hora)
        cols_feat = [c for c in df_features.columns if c not in ['atividade', 'participant_id']]
        X_all_for_scaler = df_features[cols_feat].values
        
        final_scaler = StandardScaler()
        final_scaler.fit(X_all_for_scaler)
        
        # 2. Procurar um ficheiro CSV real para teste
        print(" -> A procurar um ficheiro raw (CSV) para extrair um segmento...")
        arquivo_teste = None
        for root, dirs, files in os.walk(path):
            for f in files:
                # Tenta encontrar dados do 'part1' (Participante 1)
                if f.endswith(".csv") and "part1" in f: 
                    arquivo_teste = os.path.join(root, f)
                    break
            if arquivo_teste: break
        
        if arquivo_teste:
            print(f" -> Ficheiro encontrado: {os.path.basename(arquivo_teste)}")
            # Carregar dados brutos (N linhas, 12 colunas)
            dados_raw_completo = carregar_pessoa(arquivo_teste)
            
            # 3. Encontrar um segmento de 'Walking' (Atividade 4)
            target_ativ = 4
            win_size = 250 # 5 segundos a 50Hz (aprox 256 linhas pedidas, usamos 250 para bater certo com features)
            segmento_real = None
            
            print(f" -> A procurar segmento de atividade {target_ativ} (Walking)...")
            
            # Percorre o ficheiro com passo de 100
            for i in range(0, len(dados_raw_completo) - win_size, 100):
                window = dados_raw_completo[i : i+win_size]
                # A coluna 11 tem o Label da atividade
                labels = window[:, 11]
                
                # Se todos os pontos na janela forem da atividade alvo
                if np.all(labels == target_ativ):
                    # Sucesso! Copiar as colunas dos sensores (índices 1 a 9)
                    # Colunas: 1-3 (Acc), 4-6 (Gyr), 7-9 (Mag)
                    segmento_real = window[:, 1:10]
                    print(f" -> Segmento encontrado nas linhas {i} a {i+win_size}!")
                    break
            
            if segmento_real is not None:
                # 4. Chamar a função de deployment "chave-na-mão"
                print(" -> A executar deploy_system()...")
                predicao = deploy_system(segmento_real, final_scaler, model_final)
                
                print("-" * 40)
                print(f" RESULTADO DO SISTEMA: {predicao}")
                print(f" ATIVIDADE REAL:       Walking (4)")
                print("-" * 40)
                
                if "4" in str(predicao) or "Walking" in str(predicao):
                    print(" >> SUCESSO! A previsão está correta.")
                else:
                    print(" >> A previsão falhou (o que pode acontecer, dado Acc ~60%).")
            else:
                print(" -> Não foi encontrado nenhum segmento limpo de 'Walking' neste ficheiro.")
        else:
            print(" -> ERRO: Ficheiro de teste não encontrado no caminho indicado.")

if __name__ == "__main__":
    run_full_evaluation(n_repeats=3)