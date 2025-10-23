import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kstest, f_oneway, kruskal
import scipy as sp
import csv
import os

# --- ex 1 e 2
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

# --- 3.1
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

# --- 3.2: Função que calcula quantos outliers em cada atividade e as duas densidades = IQR
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

# --- 3.3. Função que calcula os outliers usando o método do Z-Score
def Z_Score(valores,k=1.5):
    media = np.mean(valores)
    desvio = np.std(valores)
    if desvio == 0:
        return np.zeros_like(valores, dtype=bool)
    z_scores = (valores - media) / desvio
    outliers = np.abs(z_scores) > k
    return outliers

# --- 3.4. Funcão para Outliers a vermelho e restantes azul 
def plot_outliers_varios_k(valores, atividades, ks=[3,3.5,4], titulo_base='Sensor'):
    """
    Cria um plot com 3 subplots, cada um usando um valor de k diferente para calcular outliers por atividade.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    ativs = np.unique(atividades)

    for i, k in enumerate(ks):
        ax = axes[i]
        for at in ativs:
            mask = atividades == at
            vals_atividade = valores[mask]

            # Z-score por atividade
            media = np.mean(vals_atividade)
            desvio = np.std(vals_atividade)
            if desvio == 0:
                outliers = np.zeros_like(vals_atividade, dtype=bool)
            else:
                z_scores = (vals_atividade - media) / desvio
                outliers = np.abs(z_scores) > k

            # Plota valores normais
            ax.scatter([at]*np.sum(~outliers), vals_atividade[~outliers],
                       color='blue', alpha=0.6, s=10)
            # Plota outliers
            ax.scatter([at]*np.sum(outliers), vals_atividade[outliers],
                       color='red', alpha=0.8, s=10)

        ax.set_title(f'k = {k}')
        ax.set_xlabel('Atividade')
        ax.set_xticks(ativs)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[0].set_ylabel('Módulo')
    fig.suptitle(f'{titulo_base} - Outliers por atividade', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




# --- 3.6 Função para calcular o k-means para n valores clusterns
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
    
    """# 5. Plot opcional — mostrar clusters
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(valores)), valores, c=labels, cmap='viridis', s=10)
    plt.hlines(centroids, xmin=0, xmax=len(valores), colors='red', linestyles='--', label='Centroides')
    plt.title(f"K-Means (n_clusters={n_clusters})")
    plt.xlabel("Índice da amostra")
    plt.ylabel("Módulo")
    plt.legend()
    plt.show()"""

    return labels,centroids

# --- 3.7. calculo k_means 3D
def kmeans_outliers_3d(data,n_clusters):
    """
    Aplica K-Means em dados 3D e identifica outliers (cluster mais pequeno).
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Determinar o cluster mais pequeno (potenciais outliers)
    unique, counts = np.unique(labels, return_counts=True)
    smallest_cluster = unique[np.argmin(counts)]
    outliers = labels == smallest_cluster

    return outliers, labels, centroids 

# --- 3.7. Função para plot 3D dos outliers
def plot_3d_outliers(dados_xyz, outliers, titulo):
    """
    Mostra gráfico 3D dos pontos normais (azul) e outliers (vermelho).
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dados_xyz[~outliers, 0], dados_xyz[~outliers, 1], dados_xyz[~outliers, 2], c='blue', label='Normal', alpha=0.5)
    ax.scatter(dados_xyz[outliers, 0], dados_xyz[outliers, 1], dados_xyz[outliers, 2], c='red', label='Outliers', alpha=0.8)
    ax.set_title(titulo)
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_zlabel('Eixo Z')
    ax.legend()
    plt.show() 

def outliers_kmeans(valores, n_clusters):
    labels, centroids = k_means(valores, n_clusters=n_clusters)
    
    # Identifica o cluster mais pequeno = provável "outlier"
    cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
    outlier_cluster = np.argmin(cluster_sizes)
    
    # Máscara de outliers
    outliers = labels == outlier_cluster
    return outliers, centroids, cluster_sizes

# --- 4.1 Função para verificar normalidade com teste KS
def verificar_normalidade(valores):
    stat, p = kstest((valores - np.mean(valores)) / np.std(valores), 'norm')
    return p > 0.05  # True se normal

# --- 4.1 Função para comparar atividades
def comparar_atividades(valores, atividades, sensor_nome):
    # Agrupar por atividade
    grupos = [valores[atividades == a] for a in np.unique(atividades)]

    # Verifica normalidade (para o primeiro grupo como exemplo)
    normal = all(verificar_normalidade(g) for g in grupos)
    print(f"\nSensor: {sensor_nome}")
    print("Distribuição normal?" , "Sim" if normal else "Não")

    # Escolhe o teste adequado
    if normal:
        stat, p = f_oneway(*grupos)  # ANOVA
        print(f"ANOVA -> estatística={stat:.4f}, p-valor={p:.4e}")
    else:
        stat, p = kruskal(*grupos)  # Kruskal-Wallis
        print(f"Kruskal-Wallis -> estatística={stat:.4f}, p-valor={p:.4e}")

    if p < 0.05:
        print("Diferenças significativas entre atividades.")
    else:
        print("Não há diferenças significativas detectadas.")


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

    # ---3.1 à 3.3 Boxplot com calculo do Z-score
    """ati = np.unique(atividades)
    for ativi in ati:
        val_ac = Z_Score(md_ac[atividades == ativi],k = 3)
        val_gi = Z_Score(md_gi[atividades == ativi],k = 3.5)
        val_ma = Z_Score(md_ma[atividades == ativi],k = 4)
        print(f'\nAtividade {ativi}:')
        print(f'  Acelerómetro: {np.sum(val_ac)} outliers em {len(val_ac)} ({(np.sum(val_ac)/len(val_ac))*100:.2f}%)')
        print(f'  Giroscópio: {np.sum(val_gi)} outliers em {len(val_gi)} ({(np.sum(val_gi)/len(val_gi))*100:.2f}%)')
        print(f'  Magnetómetro: {np.sum(val_ma)} outliers em {len(val_ma)} ({(np.sum(val_ma)/len(val_ma))*100:.2f}%)\n')
    """
    # Criar boxplots
    #boxplot(md_ac,atividades,'Módulo do Acelerómetro')
    #boxplot(md_gi,atividades,'Módulo do Giroscópio')
    #boxplot(md_ma,atividades,'Módulo do Magnetómetro')

    # -- 3.4 Boxplot mas com as atividades no mesmo gráfico
    # Plot para Acelerómetro
    plot_outliers_varios_k(md_ac, atividades, ks=[3,3.5,4], titulo_base='Acelerómetro')
    # Plot para Giroscópio
    plot_outliers_varios_k(md_gi, atividades, ks=[3,3.5,4], titulo_base='Giroscópio')
    # Plot para Magnetómetro
    plot_outliers_varios_k(md_ma, atividades, ks=[3,3.5,4], titulo_base='Magnetómetro')
    # --- 3.4 Boxplot que apresnta os outliers a vermelho e o restante a azul
    """ for atividade in np.unique(atividades):
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
    """
    # --- 3.6: Aplicar K-Means aos módulos
    #print("\n===== K-MEANS =====")

    """for n in [2, 3, 4]:
        print(f"\n---- K-MEANS com {n} clusters ----")
        out_ac, cent_ac, sizes_ac = outliers_kmeans(md_ac, n_clusters=n)
        print(f"Acelerómetro: {np.sum(out_ac)} outliers ({np.sum(out_ac)/len(out_ac)*100:.2f}%) - Tamanhos: {sizes_ac}")
        out_gi, cent_gi, sizes_gi = outliers_kmeans(md_gi, n_clusters=n)
        print(f"Giroscópio:   {np.sum(out_gi)} outliers ({np.sum(out_gi)/len(out_gi)*100:.2f}%) - Tamanhos: {sizes_gi}")
        out_ma, cent_ma, sizes_ma = outliers_kmeans(md_ma, n_clusters=n)
        print(f"Magnetómetro: {np.sum(out_ma)} outliers ({np.sum(out_ma)/len(out_ma)*100:.2f}%) - Tamanhos: {sizes_ma}")
        # Salva os centroides da ultima iteração
        centroids_ac = cent_ac
        centroids_gi = cent_gi
        centroids_ma = cent_ma

    print("Centroides Acelerómetro:", centroids_ac.ravel())
    print("Centroides Giroscópio:", centroids_gi.ravel())
    print("Centroides Magnetómetro:", centroids_ma.ravel())
    """

    """# --- 3.7: Aplicar K-Means no espaço 3D dos sensores ---
    sensores = {
        "Acelerómetro": dados[:, 1:4],
        "Giroscópio": dados[:, 4:7],
        "Magnetómetro": dados[:, 7:10]
    }
    for nome, valores in sensores.items():
        print(f"\n===== {nome.upper()} =====")
        for n in [2, 3, 4]:
            outliers, labels, centroids = kmeans_outliers_3d(valores, n_clusters=n)
            perc = np.sum(outliers) / len(outliers) * 100
            print(f"  n_clusters={n}: {np.sum(outliers)} outliers ({perc:.2f}%)")

            plot_3d_outliers(valores, outliers, f"{nome} - KMeans com {n} clusters")
    """
    """# 4.1 — Teste estatístico
    print("\n===== 4.1 — Análise Estatística =====")
    comparar_atividades(md_ac, atividades, "Acelerómetro")
    comparar_atividades(md_gi, atividades, "Giroscópio")
    comparar_atividades(md_ma, atividades, "Magnetómetro")"""
    
if __name__ == "__main__":
    main()