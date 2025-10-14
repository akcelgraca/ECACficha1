import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import csv
import os

# ex 1 e 2
def carregar_pessoa(ficheiro,delimiter = ',', skip_header = 1):
    dados = []
    # abre o ficheiro e lê
    with open(ficheiro,'r',newline='',encoding='utf-8') as f:
        ler = csv.reader(f,delimiter=delimiter)

        # Se houver cabeçalho, ignora a primeira linha
        if skip_header:
            next(ler,None)
        # Passa para float
        for linha in ler:
            dados.append([float(x) for x in linha])
    # transforma em array numpy
    return np.array(dados)

# ex 3.1
def cal_modulos(dados):
    #Acelerometeo = colunas 1,2,3
    md_ac = np.sqrt(dados[:,1]**2 + dados[:,2]**2 + dados[:,3]**2)
    #Giroscopio = colunas 4,5,6
    md_gi = np.sqrt(dados[:,4]**2 + dados[:,5]**2 + dados[:,6]**2)
    #Magnetometro = colunas 7,8,9
    md_ma = np.sqrt(dados[:,7]**2 + dados[:,8]**2 + dados[:,9]**2)
    return md_ac,md_gi,md_ma

def boxplot(valores,atividades,titulo):
    # Encontrar atividades únicas
    ativ = np.unique(atividades)
    dados_plot = [valores[atividades == a] for a in ativ]
    plt.figure(figsize=(10, 6))
    plt.boxplot(dados_plot, labels=ativ)
    plt.title(titulo)
    plt.xlabel('Atividades')
    plt.ylabel('Módulos')   
    plt.show()
     
#3.2: Função que calcula quantos outliers em cada atividade e as duas densidades = IQR  
def Cal_outliers(valores):
    Q1,Q3 = np.percentile(valores,[25,75])
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    # Identificar outliers
    outilers = (valores < lim_inf) | (valores > lim_sup)
    no = np.sum(outilers)
    nr = len(valores)
    d = (no/nr)*100 if nr > 0 else 0
    return d,no,nr

#3.3. Função que calcula os outliers usando o método do Z-Score
def Z_Score(valores,k=1.5):
    media = np.mean(valores)
    desvio = np.std(valores)
    if desvio == 0:
        return np.zeros_like(valores, dtype=bool)
    z_scores = (valores - media) / desvio
    outliers = np.abs(z_scores) > k
    return outliers

#3.4. Funcão para Outliers a vermelho e restantes azul 
def plot_outliers(valores,outliers,titulo):
    plt.figure(figsize=(10,5))
    plt.scatter(range(len(valores)),valores, c='blue', label='Normal')
    plt.scatter(np.where(outliers)[0],valores[outliers], c='red',label='Outliers')
    plt.title(titulo)
    plt.xlabel('Índice de amostra')
    plt.ylabel('Módulo')
    plt.legend()
    plt.show()

# 3.6 Função para calcular o k-means para n valores clusterns
def k_means(valores,n_clusters,max_iter=100):
    valores = valores.reshape(-1,1)
    indices = np.random.choice(len(valores),n_clusters,replace=False)
    centroids = valores[indices]

    for _ in range(max_iter):
        distancias = np.abs(valores - centroids.T)
        labels = np.argmin(distancias,axis=1)
        novos_centroids = np.array([
            valores[labels == i].mean() if np.any(labels == i) else centroids[i]
            for i in range(n_clusters)
        ])

        if np.allclose(centroids,novos_centroids,atol=1e-5):
            break
        centroids = novos_centroids

    # 5. Plot opcional — mostrar clusters
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(valores)), valores, c=labels, cmap='viridis', s=10)
    plt.hlines(centroids, xmin=0, xmax=len(valores), colors='red', linestyles='--', label='Centroides')
    plt.title(f"K-Means (n_clusters={n_clusters})")
    plt.xlabel("Índice da amostra")
    plt.ylabel("Módulo")
    plt.legend()
    plt.show()

    return labels,centroids



def main():
    pasta_principal = "C:\\Users\\akcel\\OneDrive\\Ambiente de Trabalho\\school\\FCTUC\\UC_25-26\\Engenharia de Características para Aprendizagem Computacional\\Prática\\TP1"
    todos_dados = []

    # percorre todas as subpastas (part0, part1, etc.)
    for root, dirs, files in os.walk(pasta_principal):
        for f in files:
            if f.endswith('.csv'):
                caminho = os.path.join(root, f)
                dados = carregar_pessoa(caminho)
                todos_dados.append(dados)

    dados = np.vstack(todos_dados)
    md_ac,md_gi,md_ma = cal_modulos(dados)
    atividades = dados[:,11].astype(int)
    k_ac,k_gi,k_ma = 3,3.5,4
    
    # 3.1 à 3.3 Boxplot com calculo do Z-score
    ati = np.unique(atividades)
    for ativi in ati:
        val_ac = Z_Score(md_ac[atividades == ativi],k = 3)
        val_gi = Z_Score(md_gi[atividades == ativi],k = 3.5)
        val_ma = Z_Score(md_ma[atividades == ativi],k = 4)
        print(f'\nAtividade {ativi}:')
        print(f'  Acelerómetro: {np.sum(val_ac)} outliers em {len(val_ac)} ({(np.sum(val_ac)/len(val_ac))*100:.2f}%)')
        print(f'  Giroscópio: {np.sum(val_gi)} outliers em {len(val_gi)} ({(np.sum(val_gi)/len(val_gi))*100:.2f}%)')
        print(f'  Magnetómetro: {np.sum(val_ma)} outliers em {len(val_ma)} ({(np.sum(val_ma)/len(val_ma))*100:.2f}%)\n')

    # Criar boxplots
    boxplot(md_ac,atividades,'Módulo do Acelerómetro')
    boxplot(md_gi,atividades,'Módulo do Giroscópio')
    boxplot(md_ma,atividades,'Módulo do Magnetómetro')
    
    # 3.4 Boxplot que apresnta os outliers a vermelho e o restante a azul
    for atividade in np.unique(atividades):
        ac_vals = md_ac[atividades == atividade]
        gi_vals = md_gi[atividades == atividade]
        ma_vals = md_ma[atividades == atividade]

        out_ac = Z_Score(ac_vals, k_ac)
        out_gi = Z_Score(gi_vals, k_gi)
        out_ma = Z_Score(ma_vals, k_ma)

        print(f"\nAtividade {atividade}:")
        print(f"  Acelerómetro: {np.sum(out_ac)} outliers ({(np.sum(out_ac)/len(out_ac))*100:.2f}%)")
        print(f"  Giroscópio:   {np.sum(out_gi)} outliers ({(np.sum(out_gi)/len(out_gi))*100:.2f}%)")
        print(f"  Magnetómetro: {np.sum(out_ma)} outliers ({(np.sum(out_ma)/len(out_ma))*100:.2f}%)")

        # Plot dos valores com outliers marcados
        plot_outliers(ac_vals, out_ac, f'Acelerómetro - Atividade {atividade} (k={k_ac})')
        plot_outliers(gi_vals, out_gi, f'Giroscópio - Atividade {atividade} (k={k_gi})')
        plot_outliers(ma_vals, out_ma, f'Magnetómetro - Atividade {atividade} (k={k_ma})')
    
    # 3.6: Aplicar K-Means aos módulos
    print("\n===== K-MEANS =====")

    centroids_ac = k_means(md_ac, n_clusters=3)
    centroids_gi = k_means(md_gi, n_clusters=3)
    centroids_ma = k_means(md_ma, n_clusters=3)

    print("Centroides Acelerómetro:", centroids_ac.ravel())
    print("Centroides Giroscópio:", centroids_gi.ravel())
    print("Centroides Magnetómetro:", centroids_ma.ravel())
    
if __name__ == "__main__":
    main()