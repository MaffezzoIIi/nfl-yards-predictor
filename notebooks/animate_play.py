import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
import numpy as np
import sys
import os

# Adiciona src/ ao path para importar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import prepare_data
from src.data_loader import load_plays

# Carrega os dados
tracking = pd.read_csv("data/tracking_week_1.csv")
plays = load_plays()

# Escolhe uma jogada válida
sample_play = tracking[['gameId', 'playId']].drop_duplicates().sample(1).iloc[0]
gameId = sample_play['gameId']
playId = sample_play['playId']
df = tracking[(tracking['gameId'] == gameId) & (tracking['playId'] == playId)]

# Prepara os dados para predição
X_pred, _ = prepare_data(df.copy(), plays.copy())
model = joblib.load("models/trained_model.pkl")
predicted_yards = model.predict(X_pred)[0]

# Valor real da jogada
real_yards = plays[
    (plays['gameId'] == gameId) & (plays['playId'] == playId)
]['yardsGained'].values[0]

# Direção da jogada
play_direction = df['playDirection'].iloc[0]
flip = True if play_direction == 'left' else False
teams = df['club'].unique()
colors = {team: color for team, color in zip(teams, ['blue', 'red', 'green', 'orange', 'purple'])}

# Função de animação
def animate_play(play_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)

    def update(frame):
        ax.clear()
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        ax.set_title(
            f"Play {playId} | Real: {real_yards} jardas – Predito: {predicted_yards:.2f} jardas",
            fontsize=14
        )

        frame_data = play_df[play_df['frameId'] == frame]
        x = frame_data['x']
        y = frame_data['y']
        if flip:
            x = 120 - x
            y = 53.3 - y

        colors_list = [colors.get(club, 'gray') for club in frame_data['club']]
        scatter = ax.scatter(x, y, c=colors_list, s=300)

        texts = []
        for _, row in frame_data.iterrows():
            if pd.notna(row['jerseyNumber']):
                x_val = 120 - row['x'] if flip else row['x']
                y_val = 53.3 - row['y'] if flip else row['y']
                text = ax.text(x_val, y_val, str(int(row['jerseyNumber'])), color='white',
                               ha='center', va='center', fontsize=8, fontweight='bold')
                texts.append(text)
        return [scatter] + texts

    frames = sorted(play_df['frameId'].unique())
    anim = FuncAnimation(fig, update, frames=frames, interval=100)
    plt.xlabel("Campo (jardas)")
    plt.ylabel("Largura do campo")
    plt.tight_layout()
    plt.show()

# Executa animação
animate_play(df)
