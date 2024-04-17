import pickle

def load_models(model_path="models/linear_regressor.pkl",
               scaler_x_path="models/min_max_scaler_x.pkl",
               scaler_y_path="models/min_max_scaler_y.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_x_path, "rb") as f:
        scaler_x = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)
    return model, scaler_x, scaler_y

def predict(density, model):
    result = model.predict(density)
    return result