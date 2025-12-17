"""
Akcel Soares da Graça - 2022241055
Edson Alage - 2021244423
Yel - 2023250149
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kstest, f_oneway, kruskal
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.fft import rfft, rfftfreq
from sklearn.cluster import KMeans
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

    return labels,centroids

# --- 3.7. calculo k_means 3D

from sklearn.preprocessing import StandardScaler

def kmeans_outliers_detection(md_ac, md_ma, md_gi, n_clusters=4, threshold_factor=7.0, plot_centroids=True):
    """
    Detecta outliers no espaço 3D (Acelerómetro=X, Giroscópio=Z, Magnetómetro=Y)
    usando KMeans no espaço normalizado. Outliers são identificados por cluster:
    ponto é outlier se distância ao centróide > mean_cluster + threshold_factor * std_cluster.

    Retorna: labels, centroids (no espaço escalado), outliers_mask (booleano).
    """
    # Montar dados 3D na ordem correta: X=ac, Y=ma, Z=gi
    data_3d = np.vstack([md_ac, md_ma, md_gi]).T  # note: essa é a ordem X,Y,Z para plot
    # Normalizar para igualar escalas
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_3d)

    # KMeans no espaço normalizado
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    centroids = kmeans.cluster_centers_

    # Distâncias de cada ponto ao seu centróide (no espaço escalado)
    distances = np.linalg.norm(data_scaled - centroids[labels], axis=1)

    # Identificar outliers por cluster (limiar baseado em mean + factor * std do cluster)
    outliers_mask = np.zeros(len(distances), dtype=bool)
    for c in range(n_clusters):
        mask_c = labels == c
        if np.sum(mask_c) == 0:
            continue
        d_c = distances[mask_c]
        lim_c = np.mean(d_c) + threshold_factor * np.std(d_c)
        outliers_mask[mask_c] = d_c > lim_c

    # Estatísticas
    total = len(distances)
    n_out = int(np.sum(outliers_mask))
    print(f"\nKMeans 3D: clusters={n_clusters} | outliers={n_out}/{total} ({n_out/total*100:.2f}%)")

    # Plot 3D: azul = normal, vermelho = outlier
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(md_ac[~outliers_mask], md_ma[~outliers_mask], md_gi[~outliers_mask],
               c='blue', alpha=0.6, label='Normal', s=10)
    ax.scatter(md_ac[outliers_mask], md_ma[outliers_mask], md_gi[outliers_mask],
               c='red', alpha=0.9, label='Outlier', s=20)

    # Opcional: marcar centróides (transformar centróides de volta ao espaço original para plot)
    if plot_centroids:
        centroids_orig = scaler.inverse_transform(centroids)  # retornar ao espaço original
        ax.scatter(centroids_orig[:,0], centroids_orig[:,1], centroids_orig[:,2],
                   c='black', marker='X', s=80, label='Centroides')

    ax.set_xlabel('Acelerómetro (módulo)')
    ax.set_ylabel('Magnetómetro (módulo)')
    ax.set_zlabel('Giroscópio (módulo)')
    ax.set_title(f'Deteção de Outliers com KMeans (k={n_clusters}, factor={threshold_factor})')
    ax.legend()
    plt.show()

    return labels, centroids, outliers_mask



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

# 4.2 extração de features
# ======= TEMPORAL FEATURES =======
def extract_temporal_features(signal):
    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "var": np.var(signal),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "amplitude": np.max(signal) - np.min(signal),
        "mean_abs": np.mean(np.abs(signal)),
        "zero_cross": ((signal[:-1] * signal[1:]) < 0).sum(),
        "energy": np.sum(signal ** 2) / len(signal)
    }
    return features

# ======= SPECTRAL FEATURES =======
def extract_spectral_features(signal, fs):
    fft_vals = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), 1/fs)
    psd = fft_vals ** 2
    
    # Normalizar PSD para probabilidades
    psd_norm = psd / np.sum(psd)
    
    features = {
        "spectral_energy": np.mean(psd),
        "spectral_entropy": entropy(psd_norm),
        "dominant_freq": freqs[np.argmax(fft_vals)],
        "spectral_centroid": np.sum(freqs * psd_norm),
        "spectral_spread": np.sqrt(np.sum(((freqs - np.sum(freqs * psd_norm))**2) * psd_norm))
    }
    return features

# ======= PHYSICAL FEATURES (Mi Zhang & Sawchuk, 2011) =======
def extract_physical_features(window_acc, window_gyr, fs=50):
    """
    Calcula features físicas a partir de uma janela de 5 s dos sinais de aceleração e giroscópio.
    window_acc e window_gyr são arrays Nx3 (x,y,z)
    """
    # Módulo de aceleração e giroscópio
    acc_mag = np.linalg.norm(window_acc, axis=1)
    gyr_mag = np.linalg.norm(window_gyr, axis=1)

    # AI e VI — média e variância do módulo da aceleração
    AI = np.mean(acc_mag)
    VI = np.var(acc_mag)

    # SMA — Signal Magnitude Area
    SMA = np.mean(np.abs(window_acc[:, 0]) + np.abs(window_acc[:, 1]) + np.abs(window_acc[:, 2]))

    # EVA1, EVA2 — 2 maiores autovalores da matriz de covariância dos 3 eixos
    cov_acc = np.cov(window_acc.T)
    eigvals = np.linalg.eigvals(cov_acc)
    EVA1, EVA2 = np.sort(np.real(eigvals))[-2:]

    # CAGH — correlação entre eixo gravitacional (x) e horizontal (sqrt(y²+z²))
    horiz_mag = np.sqrt(window_acc[:, 1]**2 + window_acc[:, 2]**2)
    if np.std(window_acc[:, 0]) == 0 or np.std(horiz_mag) == 0:
        CAGH = 0
    else:
        CAGH = np.corrcoef(window_acc[:, 0], horiz_mag)[0, 1]

    # AVH / AVG — velocidade média horizontal e vertical (via integração simples)
    dt = 1 / fs
    vel = np.cumsum(window_acc * dt, axis=0)
    horiz_vel = np.sqrt(vel[:, 1]**2 + vel[:, 2]**2)
    vert_vel = np.abs(vel[:, 0])
    AVH = np.mean(horiz_vel)
    AVG = np.mean(vert_vel)

    # ARATG — ângulo de rotação acumulado em torno da gravidade (giroscópio)
    rot_ang = np.cumsum(gyr_mag * dt)
    ARATG = rot_ang[-1] - rot_ang[0]

    # AAE / ARE — energia média e relativa (aceleração e giroscópio)
    AAE = np.sum(acc_mag**2) / len(acc_mag)
    ARE = np.sum(gyr_mag**2) / len(gyr_mag)

    # DF — frequência dominante do módulo da aceleração
    fft_vals = np.abs(rfft(acc_mag))
    freqs = rfftfreq(len(acc_mag), 1/fs)
    DF = freqs[np.argmax(fft_vals)]

    return {
        "AI": AI,
        "VI": VI,
        "SMA": SMA,
        "EVA1": EVA1,
        "EVA2": EVA2,
        "CAGH": CAGH,
        "AVH": AVH,
        "AVG": AVG,
        "ARATG": ARATG,
        "AAE": AAE,
        "ARE": ARE,
        "DF": DF
    }

def extract_features_all(data, atividades, fs=50, window_sec=5, overlap=0.5):
    win_size = int(window_sec * fs)
    step = int(win_size * (1 - overlap))

    features_list = []
    labels_list = []

    for start in range(0, len(data) - win_size, step):
        end = start + win_size
        janela = data[start:end, :]  # assume colunas [ax, ay, az, gx, gy, gz, mx, my, mz]
        atv_window = atividades[start:end]

        # Verifica se toda a janela pertence à mesma atividade
        if len(np.unique(atv_window)) != 1:
            continue
        
        atividade = atv_window[0]
        feat_dict = {"atividade": atividade}

        # Para cada eixo de cada sensor
        sensores = ["acc", "gyr", "mag"]
        for i, sensor in enumerate(sensores):
            base_idx = i * 3
            for eixo, label in zip(range(3), ['x', 'y', 'z']):
                sinal = janela[:, base_idx + eixo]

                # Temporais
                t_feats = extract_temporal_features(sinal)
                feat_dict.update({f"{sensor}_{label}_{k}": v for k, v in t_feats.items()})

                # Espectrais
                f_feats = extract_spectral_features(sinal, fs)
                feat_dict.update({f"{sensor}_{label}_{k}": v for k, v in f_feats.items()})
                # === Features físicas (acel + giro em 3D) ===
                acc_window = janela[:, 0:3]
                gyr_window = janela[:, 3:6]
                phys_feats = extract_physical_features(acc_window, gyr_window, fs)
                feat_dict.update(phys_feats)
    
        features_list.append(feat_dict)
        labels_list.append(atividade)

    return pd.DataFrame(features_list)

# --- 4.3 Função para aplicar PCA
def aplicar_pca(features_df, n_componentes=None, plot=True):
    """
    Aplica PCA ao feature set e mostra a variância explicada.
    - features_df: DataFrame com features (sem colunas de labels)
    - n_componentes: número de componentes principais (None usa todas)
    - plot: se True, mostra gráfico da variância explicada
    """
    # Separa as features e o label
    X = features_df.drop(columns=['atividade'])
    y = features_df['atividade'].values

    # Normaliza as features (média=0, desvio=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplica PCA
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X_scaled)

    # Variância explicada
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)

    print("\n=== Análise PCA ===")
    print(f"Número de componentes: {pca.n_components_}")
    print(f"Variância total explicada (1ª componente): {var_exp[0]*100:.2f}%")
    print(f"Variância acumulada (primeiras 5 componentes): {cum_var_exp[4]*100:.2f}%")
    print(f"Variância acumulada total: {cum_var_exp[-1]*100:.2f}%")

    # Gráfico da variância explicada
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(var_exp)+1), cum_var_exp*100, marker='o')
        plt.xlabel('Número de Componentes Principais')
        plt.ylabel('Variância Explicada (%)')
        plt.title('PCA - Variância Explicada Acumulada')
        plt.grid(True)
        plt.show()

    # Retorna o dataset transformado
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    X_pca_df['atividade'] = y

    return X_pca_df, pca

def fisher_score(X, y):
    """
    Calcula o Fisher Score de cada feature num dataset supervisionado.
    X: ndarray (n_amostras x n_features)
    y: vetor de labels
    Retorna: array com Fisher Scores
    """
    classes = np.unique(y)
    n_features = X.shape[1]

    scores = np.zeros(n_features)
    overall_mean = np.mean(X, axis=0)

    for i in range(n_features):
        num = 0.0
        den = 0.0
        for c in classes:
            X_c = X[y == c, i]
            mean_c = np.mean(X_c)
            var_c = np.var(X_c)
            n_c = len(X_c)
            num += n_c * (mean_c - overall_mean[i])**2
            den += n_c * var_c
        scores[i] = num / (den + 1e-8)  # evita divisão por zero

    return scores


def reliefF(X, y, n_neighbors=10):
    """
    Implementação simplificada do algoritmo ReliefF.
    Retorna o score de importância para cada feature.
    """
    n_samples, n_features = X.shape
    classes = np.unique(y)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    scores = np.zeros(n_features)

    for i in range(n_samples):
        Xi = X[i]
        yi = y[i]
        # Ignorar o próprio ponto
        neighbors = indices[i, 1:]
        same_class = [j for j in neighbors if y[j] == yi]
        diff_class = [j for j in neighbors if y[j] != yi]

        for f in range(n_features):
            hit_diff = np.mean(np.abs(Xi[f] - X[same_class, f])) if same_class else 0
            miss_diff = np.mean(np.abs(Xi[f] - X[diff_class, f])) if diff_class else 0
            scores[f] += miss_diff - hit_diff

    return scores / n_samples


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

    """# -- 3.4 versão 2.0 Boxplot mas com as atividades no mesmo gráfico
    # Plot para Acelerómetro
    plot_outliers_varios_k(md_ac, atividades, ks=[3,3.5,4], titulo_base='Acelerómetro')
    # Plot para Giroscópio
    plot_outliers_varios_k(md_gi, atividades, ks=[3,3.5,4], titulo_base='Giroscópio')
    # Plot para Magnetómetro
    plot_outliers_varios_k(md_ma, atividades, ks=[3,3.5,4], titulo_base='Magnetómetro')"""
    
    # --- 3.4. versão 1.0 Boxplot que apresnta os outliers a vermelho e o restante a azul
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

    # --- 4.7. K-Means 3D nos três módulos
    """print("3.6-3.7 Aplicando K-means para deteção de outliers...")
    labels_3d, centroids_3d, outliers_3d = kmeans_outliers_detection(md_ac, md_ma, md_gi, n_clusters=4)"""

    # Estatísticas por cluster
    """unique, counts = np.unique(labels_3d, return_counts=True)
    print("\nDistribuição por cluster:")
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} pontos")
    """

    # data: array Nx9 (ax, ay, az, gx, gy, gz, mx, my, mz)

    """# 4.1 — Teste estatístico
    print("\n===== 4.1 — Análise Estatística =====")
    comparar_atividades(md_ac, atividades, "Acelerómetro")
    comparar_atividades(md_gi, atividades, "Giroscópio")
    comparar_atividades(md_ma, atividades, "Magnetómetro")

    # --- 4.2: Extração de features
    # atividades: vetor Nx1 com os rótulos das atividades
    features_df = extract_features_all(dados[:, :9], dados[:, 11].astype(int), fs=50)
    pd.set_option('display.max_columns', None)  # mostra todas as colunas
    pd.set_option('display.width', 200)         # evita quebra de linha automática
    #print(features_df.head())

    # --- 4.3: PCA ---
    print("\n===== 4.3 — PCA =====")
    features_df = features_df.fillna(features_df.mean())
    X_pca_df, pca_model = aplicar_pca(features_df, n_componentes=None, plot=True)


    # Visualizar as duas primeiras componentes principais
    plt.figure(figsize=(8,6))
    for atividade in np.unique(X_pca_df['atividade']):
        subset = X_pca_df[X_pca_df['atividade'] == atividade]
        plt.scatter(subset['PC1'], subset['PC2'], s=15, label=f'Atividade {atividade}', alpha=0.7)
    plt.title('PCA - Projeção nas 2 Primeiras Componentes')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 4.5 — Seleção de Features: Fisher Score e ReliefF ---
    print("\n===== 4.5 — Feature Selection =====")

    # Preparar dados (sem coluna de atividade)
    X = features_df.drop(columns=['atividade']).values
    y = features_df['atividade'].values
    feature_names = features_df.drop(columns=['atividade']).columns

    # Fisher Score
    fisher_scores = fisher_score(X, y)
    fisher_ranking = np.argsort(fisher_scores)[::-1]
    print("\nTop 10 features (Fisher Score):")
    for idx in fisher_ranking[:10]:
        print(f"{feature_names[idx]:35s} -> {fisher_scores[idx]:.4f}")

    # ReliefF
    relief_scores = reliefF(X, y, n_neighbors=10)
    relief_ranking = np.argsort(relief_scores)[::-1]
    print("\nTop 10 features (ReliefF):")
    for idx in relief_ranking[:10]:
        print(f"{feature_names[idx]:35s} -> {relief_scores[idx]:.4f}")

    # Gráficos comparativos
    plt.figure(figsize=(10,5))
    plt.bar(range(10), fisher_scores[fisher_ranking[:10]], alpha=0.7, label='Fisher Score')
    plt.bar(range(10), relief_scores[relief_ranking[:10]], alpha=0.7, label='ReliefF')
    plt.xticks(range(10), [feature_names[i] for i in fisher_ranking[:10]], rotation=90)
    plt.ylabel("Score de Importância")
    plt.title("Comparação: Fisher Score vs ReliefF (Top 10 Features)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- 4.6 — Comparação das 10 melhores features ---
    print("\n===== 4.6 — Comparação das 10 Melhores Features =====")

    # Obter as top 10 features de cada método
    top10_fisher = [feature_names[i] for i in fisher_ranking[:10]]
    top10_relief = [feature_names[i] for i in relief_ranking[:10]]

    # Mostrar as listas
    print("\nTop 10 - Fisher Score:")
    for i, feat in enumerate(top10_fisher, 1):
        print(f"{i:2d}. {feat}")

    print("\nTop 10 - ReliefF:")
    for i, feat in enumerate(top10_relief, 1):
        print(f"{i:2d}. {feat}")

    # Identificar features em comum
    common_features = set(top10_fisher).intersection(top10_relief)
    print(f"\nFeatures em comum ({len(common_features)}): {', '.join(common_features) if common_features else 'Nenhuma'}")

    # Comparar graficamente (importâncias normalizadas)
    fisher_top_vals = fisher_scores[fisher_ranking[:10]]
    relief_top_vals = relief_scores[relief_ranking[:10]]

    # Normalizar para comparar na mesma escala
    fisher_top_vals /= np.max(fisher_top_vals)
    relief_top_vals /= np.max(relief_top_vals)

    plt.figure(figsize=(10,6))
    plt.bar(np.arange(10) - 0.2, fisher_top_vals, width=0.4, label='Fisher Score', color='royalblue')
    plt.bar(np.arange(10) + 0.2, relief_top_vals, width=0.4, label='ReliefF', color='orange')
    plt.xticks(np.arange(10), top10_fisher, rotation=90)
    plt.title('Comparação das 10 Melhores Features — Fisher vs ReliefF')
    plt.ylabel('Importância Normalizada')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 4.6.1 — Obter features selecionadas para um instante específico ---

    # Escolher as top 10 do Fisher Score (ou ReliefF)
    selected_features = top10_fisher  # ou top10_relief

    # Criar novo DataFrame apenas com essas colunas
    selected_df = features_df[selected_features + ['atividade']]
    instante = 100
    print(f"\nValores das Top 10 features (Fisher Score) no instante {instante}:")
    print(selected_df.iloc[instante])
    """
if __name__ == "__main__":
    main()