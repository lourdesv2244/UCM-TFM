from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import numpy as np
#import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
from scipy import stats
from scipy.stats import norm, skew
from Funcion_Limpieza import limpieza, insertadb

app = Flask(__name__)


uploads_dir = '/instance/uploads'
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():


    modelo = open('modelo_RF_hp.pkl', 'rb')
    clf = joblib.load(modelo)


    if request.method == 'POST':
        message = request.files['fileupload']
        message.save(os.path.join(uploads_dir, secure_filename(message.filename)))
        data = pd.read_csv(message.filename)
        df_id = data["Id"]

        datos = limpieza(data)

        my_prediction = np.expm1(clf.predict(datos))

        df_id_b = pd.DataFrame(df_id, columns=['Id'])
        df_mypred = pd.DataFrame(my_prediction, columns=['pred'])
        df_id_bi = pd.concat([df_id_b, df_mypred], axis=1)
        df_id_b = df_id_bi.to_numpy()

        insertar = insertadb(df_id_bi)


    return render_template('result.html', prediction=my_prediction, tam=df_id_b)


if __name__ == '__main__':
    app.run(debug=True)


