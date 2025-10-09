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
     
"""3.2: Função que calcula quantos outliers em cada atividade e as duas densidades   """
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

def main():
    diretorio = "C:\\Users\\akcel\\OneDrive\\Ambiente de Trabalho\\school\\FCTUC\\UC_25-26\\Engenharia de Características para Aprendizagem Computacional\\Prática\\TP1\\part0"
    ficheiros = [os.path.join(diretorio,f) for f in os.listdir(diretorio) if f.endswith('.csv')]
    ficheiros.sort()

    # concatenar dados de todos os ficheiros
    todos = [carregar_pessoa(f) for f in ficheiros]
    dados = np.vstack(todos)

    md_ac,md_gi,md_ma = cal_modulos(dados)
    
    atividades = dados[:,11].astype(int)
    """3.2.
    for ativi in np.unique(atividades):
        d_ac,no_ac,nr_ac = Cal_outliers(md_ac[atividades == ativi])
        d_gi,no_gi,nr_gi = Cal_outliers(md_gi[atividades == ativi])
        d_ma,no_ma,nr_ma = Cal_outliers(md_ma[atividades == ativi])
        print(f'Atividade {ativi}:')
        print(f'  Acelerómetro: {no_ac} outliers em {nr_ac} ({d_ac:.2f}%)')
        print(f'  Giroscópio: {no_gi} outliers em {nr_gi} ({d_gi:.2f}%)')
        print(f'  Magnetómetro: {no_ma} outliers em {nr_ma} ({d_ma:.2f}%)\n')
"""
    #3.3. e 3.4: apresentar os plots em que os outiers apareçam a vermelho enquanto os restantes a azul
    ati = np.unique(atividades)
    for ativi in ati:
        val_ac = Z_Score(md_ac[atividades == ativi],k = 3)
        val_gi = Z_Score(md_gi[atividades == ativi],k = 3.5)
        val_ma = Z_Score(md_ma[atividades == ativi],k = 4)
        print(f'\nAtividade {ativi}:')
        print(f'  Acelerómetro: {np.sum(val_ac)} outliers em {len(val_ac)} ({(np.sum(val_ac)/len(val_ac))*100:.2f}%)')
        print(f'  Giroscópio: {np.sum(val_gi)} outliers em {len(val_gi)} ({(np.sum(val_gi)/len(val_gi))*100:.2f}%)')
        print(f'  Magnetómetro: {np.sum(val_ma)} outliers em {len(val_ma)} ({(np.sum(val_ma)/len(val_ma))*100:.2f}%)\n')
    # Plot dos outliers
    
    # Criar boxplots
    boxplot(md_ac,atividades,'Módulo do Acelerómetro')
    boxplot(md_gi,atividades,'Módulo do Giroscópio')
    boxplot(md_ma,atividades,'Módulo do Magnetómetro')

if __name__ == "__main__":
    main()