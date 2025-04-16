from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from datetime import datetime
import uuid

app = Flask(__name__)

# Carrega o modelo
modelo = joblib.load('modelo.joblib')

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Captura os dados do formulário
        temperature = float(request.form.get("temperature"))
        humidity = float(request.form.get("humidity"))
        light = float(request.form.get("light"))
        co2 = float(request.form.get("co2"))
        humidity_ratio = float(request.form.get("humidity_ratio"))

        # Preenche automaticamente o ID e a data
        id = str(uuid.uuid4())  # Gera um ID único aleatório
        data_hora = datetime.now().strftime('%d/%m/%y %H:%M')  # Exemplo: 15/04/25 14:32

        # Monta o vetor de entrada para o modelo
        features = np.array([[temperature, humidity, light, co2, humidity_ratio]])

        # Realiza a previsão
        previsao = modelo.predict(features)

        return render_template("index.html", previsao=previsao[0], id=id, data=data_hora)
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)