Guia de Execução

Pré-requisitos:
Para executar este projeto, é necessário ter o Python 3.8+ instalado e as seguintes bibliotecas:
-numpy
-pandas
-scipy
-matplotlib
-seaborn
-scikit-learn
-torch (PyTorch - necessário para a extração de embeddings)

Configuração dos Dados:

Antes de executar, é obrigatório configurar o caminho para a pasta onde tens os ficheiros CSV do dataset (ex: part1dev1.csv, part2dev1.csv, etc.).

1.Abre o ficheiro moduleB_main.py.

2.Localiza a variável path dentro da função run_full_evaluation (logo no início).

3.Altera o caminho para a localização da tua pasta de dados.

Como Executar:

O projeto foi centralizado num único script principal que orquestra todo o processo.

Para correr a avaliação completa (treino, teste estatístico e demonstração de deployment), executa apenas:
-python moduleB_main.py