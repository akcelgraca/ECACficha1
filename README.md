# Pipeline de Reconhecimento de Atividades Humanas (HAR)

 **Engenharia de Características para Aprendizagem Computacional (ECAC)**
 Licenciatura em Engenharia Informática/ - Universidade de Coimbra (FCTUC)

Este projeto implementa uma *pipeline* completa de Data Science para **Reconhecimento de Atividades Humanas** (HAR), utilizando o dataset *FORTH-TRACE*. O sistema abrange desde a ingestão de dados brutos de sensores inerciais até à classificação final, comparando engenharia de características manual contra representações latentes (*embeddings*) de Deep Learning.

---

## Autor

* **Akcel Soares da Graça** (2022241055) 

---

## Arquitetura do Projeto

O trabalho está dividido em dois módulos complementares:

### Parte A: Engenharia de Características
Focada na análise exploratória, limpeza e extração de informação.
* **Tratamento de Dados:** Deteção de *outliers* usando métodos univariados (Z-Score, IQR) e multivariados (K-Means).
* **Extração de Features:** Cálculo de métricas temporais, espetrais (FFT) e físicas a partir de acelerómetro, giroscópio e magnetómetro.
* **Redução de Dimensionalidade:** Implementação de PCA (Principal Component Analysis).
* **Seleção de Atributos:** Implementação dos algoritmos *Fisher Score* e *ReliefF* para identificar as características mais relevantes.

### Parte B: Machine Learning e Avaliação
Focada na classificação supervisionada das atividades 1 a 7 (ex: *Standing*, *Walking*, *Climbing Stairs*).
* **Data Augmentation:** Balanceamento de classes utilizando a técnica **SMOTE** para gerar amostras sintéticas.
* **Embeddings:** Extração automática de características usando o modelo de Deep Learning **HARNet5** (Transfer Learning) sobre dados reamostrados a 30Hz.
* **Classificação:** Modelo **k-Nearest Neighbors (kNN)** com otimização automática de hiperparâmetros.
* **Validação:** Comparação de estratégias de divisão *Within-Subject* vs. *Between-Subjects*.
* **Deployment:** Sistema "chave-na-mão" capaz de receber um segmento bruto e devolver a classificação.

---

## Instalação e Requisitos

### Pré-requisitos
O projeto requer **Python 3.8+** e as seguintes bibliotecas:

```bash
pip install numpy pandas matplotlib scipy scikit-learn torch seaborn
