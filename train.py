from src.data_loader import load_all_tracking_data, load_plays
from src.preprocess import prepare_data
from src.model import train_model, evaluate_model, save_model

from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    print("🔄 Carregando dados...")
    tracking_df = load_all_tracking_data(weeks=range(1, 9))
    plays_df = load_plays()

    print("⚙️  Preparando dados...")
    X, y = prepare_data(tracking_df, plays_df)

    print("🔀 Dividindo em treino/teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("📈 Treinando modelo...")
    model = train_model(X_train, y_train)

    print("📊 Avaliando modelo...")
    metrics = evaluate_model(model, X_test, y_test)

    print("💾 Salvando modelo...")
    os.makedirs("models", exist_ok=True)
    save_model(model, "models/trained_model.pkl")

    print("✅ Processo concluído com sucesso.")