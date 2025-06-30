import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data(tracking_df: pd.DataFrame, plays_df: pd.DataFrame) -> tuple:
    plays_df_reduced = plays_df[['gameId', 'playId', 'yardsGained']]
    merged = tracking_df.merge(plays_df_reduced, on=['gameId', 'playId'], how='inner')

    first_frames = merged.groupby(['gameId', 'playId']).first().reset_index()

    features = ['x', 'y', 's', 'a', 'dis', 'o', 'dir']
    first_frames = first_frames.dropna(subset=features + ['yardsGained'])

    X = first_frames[features]
    y = first_frames['yardsGained']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
