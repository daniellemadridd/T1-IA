# T1-IA
Este projeto consiste em um sistema de IA capaz de classificar estados de um tabuleiro de Jogo da Velha 3x3 em 4 categorias: Tem jogo, Empate, O vence e X vence.   

## Membros do Grupo
-Danielle dos Reis Madrid<br>
-Vitor Rafael Gonçalves<br>
- Francisco Cassol Raymundo

# Algoritmos usados :
Árvore de Decisão - Francisco<br>
Random Forest - Danielle<br>
K-NN - Danielle<br>
SVM - Vitor<br>
Multi Layer Percepton - Vitor<br>
    
    
# Organização do Repositório
/dados: Contém o dataset balanceado e processado<br>
/modelos: Modelos de IA treinados e exportados<br>  
/notebooks: Arquivos Jupyter (.ipynb) com todo o processo de limpeza, treinamento e validação dos algoritmos.<br>  
/src: Código do Front onde o usuário pode interagir com a IA.<br>   

 ## Organização do Repositório

- **`/datasetOriginal`**: Contém o dataset original fornecido pela professora, sem modificações.
- **`/notebook`**: Contém o Jupyter Notebook (`.ipynb`) com as etapas de análise exploratória, pré-processamento, balanceamento dos dados, treinamento e avaliação dos algoritmos.
- **`/modelos`**: Armazena os modelos treinados e exportados (`.pkl`), utilizados pela aplicação.
- **`/src`**: Contém o código da interface desenvolvida em Streamlit, permitindo ao usuário interagir com o jogo da velha e selecionar diferentes algoritmos de IA.

## Bibliotecas Utilizadas

- **Python** — Linguagem de programação utilizada no projeto.
- **Pandas** — Manipulação e análise dos dados.
- **NumPy** — Operações numéricas e processamento de dados.
- **Scikit-learn** — Implementação, treinamento e avaliação dos algoritmos k-NN, Random Forest, SVM, MLP e Árvore de Decisão.
- **Joblib** — Exportação e carregamento dos modelos treinados.
- **Streamlit** — Desenvolvimento da interface interativa do jogo da velha.
- **Matplotlib / Seaborn** — Visualização de dados e resultados dos modelos, caso utilizadas no notebook.

## Instalação das Dependências
Para instalar todas as bibliotecas necessárias para executar o projeto, basta abrir o terminal na raiz do repositório e executar:

```bash
python -m pip install -r requirements.txt
```

Esse comando instalará automaticamente todas as dependências listadas no arquivo `requirements.txt`.