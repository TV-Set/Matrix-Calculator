from flask import Flask, render_template, request
from processing import predict, load_models

app = Flask(__name__)
@app.route("/", methods=["get", "post"]) # 127.0.0.1:5000 + "/" = 127.0.0.1:5000/

def main(): # Данная функция вызывается с помощью декоратора "@app.route()"
    model, scaler_x, scaler_y = load_models()
    message = "Ничего не введено"
    if request.method == "POST":
        patch = request.form.get("patch")
        try:
            patch = float(patch)
            density = predict(scaler_x.transform([[patch]]), model)
            result = scaler_y.inverse_transform([density])
            message = f'Соотношение "матрица-наполнитель": {result[0][0]}'
        except:
            message = f"Вы ввели некорректное значение: {patch}"

    return render_template("index.html", message=message)

app.run()