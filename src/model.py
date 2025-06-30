from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(X_train, y_train):
    """
    Treina um modelo XGBoost Regressor com os dados fornecidos.
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo com métricas de regressão.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return {"mse": mse, "r2": r2}

def save_model(model, path):
    """
    Salva o modelo treinado no caminho especificado.
    """
    joblib.dump(model, path)

def load_model(path):
    """
    Carrega um modelo salvo.
    """
    return joblib.load(path)
