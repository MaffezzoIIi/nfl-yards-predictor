# src/data_loader.py

import pandas as pd
import os

DATA_DIR = "data" 

def load_tracking_data(week: int) -> pd.DataFrame:
    """Carrega dados de rastreamento para a semana especificada."""
    path = os.path.join(DATA_DIR, f"tracking_week_{week}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)

def load_players() -> pd.DataFrame:
    """Carrega dados dos jogadores."""
    path = os.path.join(DATA_DIR, "players.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)

def load_games() -> pd.DataFrame:
    """Carrega dados das partidas."""
    path = os.path.join(DATA_DIR, "games.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)

def load_plays() -> pd.DataFrame:
    """Carrega dados das jogadas."""
    path = os.path.join(DATA_DIR, "plays.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)
