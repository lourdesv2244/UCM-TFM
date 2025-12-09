import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skew
import mysql.connector

def limpieza(data):


    #PRE-PROCESSING & DATA CLEANING
        train = pd.read_csv('train.csv')
        df_id = data["Id"]
        df_id = df_id.to_numpy()
        train = train.drop(["SalePrice"], axis=1)
        junto_df = pd.concat([train, data], ignore_index=True)

        #Manejo de categoricas
        categoricas = [col for col in junto_df.columns.values if junto_df[col].dtype == 'object']
        categoricas_df = junto_df[categoricas]
        numericas_df = junto_df.drop(categoricas, axis=1)

        #Manejo de asimetria para numericas
        num_skew = numericas_df.apply(lambda x: skew(x.dropna()))
        num_skew = num_skew[num_skew > 0.75]
        numericas_df[num_skew.index] = np.log1p(numericas_df[num_skew.index])

        #Manejo de missing values
        data_len = numericas_df.shape[0]

        for col in numericas_df.columns.values:
            missing_values = numericas_df[col].isnull().sum()

            # drop si tiene mas de 50 valores valores missing
            if missing_values > 50:
                numericas_df = numericas_df.drop(col, axis=1)
            # Si es menor a 50, lo lleno con la media
            else:
                numericas_df = numericas_df.fillna(numericas_df[col].median())


        data_len = categoricas_df.shape[0]

        for col in categoricas_df.columns.values:
            missing_values = categoricas_df[col].isnull().sum()

            # drop si tiene mas de 50 valores missing
            if missing_values > 50:
                print("Eliminando columna: {}".format(col))
                categoricas_df = categoricas_df.drop(col, axis=1)

            # Si es menor a 50, lo lleno con 'No Aplica'
            else:
                categoricas_df = categoricas_df.fillna('No aplica')
                pass

        categoricas_df_dummies = pd.get_dummies(categoricas_df)
        datos = pd.concat([numericas_df, categoricas_df_dummies], axis=1)
        datos = datos.drop(["Id"], axis=1)
        datos = datos.iloc[len(train):]

        #FIN PRE-PROCESSING & DATA CLEANING

        #INSERTO EN BASE DE DATOS


        miConexion = mysql.connector.connect(host='localhost', user='root', passwd='root', db='modelo_casas')
        cur = miConexion.cursor()

        #train = pd.read_csv('test_carga_mysql.csv', index_col=False, delimiter=',')
        train = data
        train = train.fillna(0)

        for i, row in train.iterrows():
            sql = "INSERT INTO modelo_casas.casas VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,0)"
            cur.execute(sql, tuple(row))
            miConexion.commit()

        miConexion.close()

        #FIN DE INSERTAR EN BASE DE DATOS

        return datos


def insertadb(df_id_b):

    miConexion = mysql.connector.connect(host='localhost', user='root', passwd='root', db='modelo_casas')
    cur = miConexion.cursor()

    for i, row in df_id_b.iterrows():
        sql = "UPDATE modelo_casas.casas SET Prediccion = %s WHERE ID = %s"
        cur.execute(sql, (row['pred'], row['Id']))
        miConexion.commit()

    miConexion.close()

    return