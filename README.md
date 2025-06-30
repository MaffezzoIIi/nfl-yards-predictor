# 🏈 NFL Yards Predictor

Este projeto tem como objetivo utilizar os dados do NFL Big Data Bowl 2025 para **prever jardas ganhas** por um jogador em uma jogada de corrida e também **visualizar essas jogadas em animações** com integração ao modelo preditivo.

---

## 📁 Estrutura do Projeto

```text
nfl-yards-predictor/
├── data/               # Arquivos CSV da competição (tracking_week1.csv, plays.csv, etc.)
├── models/             # Modelos treinados (.pkl)
├── notebooks/          # Notebooks e scripts de visualização (ex: animate_play.py)
├── src/                # Código fonte principal
│   ├── data_loader.py  # Funções para carregar os dados
│   ├── preprocess.py   # Funções para limpar e preparar os dados
│   ├── model.py        # Treinamento, avaliação e salvamento do modelo
├── train.py            # Pipeline principal de treinamento
├── requirements.txt    # Dependências do projeto
└── README.md           # Este arquivo
```

---

## 🔧 Tecnologias Usadas

- Python 3.9+
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- joblib

Instale com:

```bash
pip install -r requirements.txt
```

---

## 🚀 Objetivos do Projeto

1. **Treinar um modelo de machine learning (XGBoost)** para prever quantas jardas um jogador irá ganhar com base em dados iniciais da jogada.
2. **Visualizar uma jogada da NFL** como uma animação com a posição dos jogadores e o número da camisa.
3. **Integrar a predição à visualização**, exibindo o valor previsto pelo modelo durante a animação.

---

## 📈 Treinamento do Modelo

Execute o treinamento com:

```bash
python train.py
```

Este script:

- Carrega os dados de `tracking_week_1.csv` e `plays.csv`
- Seleciona apenas os primeiros frames dos portadores da bola
- Gera features (posição, velocidade, direção, etc.)
- Treina um modelo de regressão XGBoost
- Salva o modelo em `models/trained_model.pkl`

---

## 🎬 Animação com Predição

Visualize uma jogada com predição integrada:

```bash
python notebooks/animate_play.py
```

Esse script:

- Seleciona uma jogada aleatória do dataset
- Usa o modelo treinado para prever as jardas
- Exibe a movimentação dos jogadores com o número da camisa
- Mostra o valor previsto no topo da tela

---

## 📌 Exemplo de visualização

```text
Play 1020 – Predição: 6.23 jardas
```

---

## 📚 Fontes dos Dados

Baixe os dados da competição em:  
[https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data)

Coloque os arquivos dentro da pasta `data/`.

---

## 🤝 Contribuição

Sinta-se à vontade para abrir *issues*, *pull requests* ou ideias de novos modelos e animações.

---

## ⚠️ Observações

- Este projeto é focado em análise exploratória e aprendizado supervisionado.
- Não usamos dados em tempo real nem qualquer conexão com estatísticas oficiais da NFL.

---

## 👨‍💻 Autor

Desenvolvido por Thomas Maffezzolli – 2025
