#Diplomado python 
#Autor: Jonnier Andres Teran Morales 
#ID No:502195
#Id:1003064599
#ID:correo:Jonnier.teran@upb.edu.co
#Cel:3245644212

#Librerias 
from email.policy import default
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score



#leemos la base de datos
dfNetflix= pd.read_excel('Netflix_list.xlsx')
dfNetflix["duracion"] = pd.to_numeric(dfNetflix['duration'].replace('([^0-9]*)','', regex=True), errors='coerce')


listaCondiciones = [
    (dfNetflix["type"] == "Movie"),
    (dfNetflix["type"] == "TV Show")
    ]
listaSelecciones = [1.0, 2.0]
dfNetflix["Cod_type"] =  np.select(listaCondiciones, listaSelecciones, default='Not Specified')

listaCondiciones = [
    (dfNetflix["duration"].str.contains("Season").astype(np.bool_)),
    (dfNetflix["duration"].str.contains("min").astype(np.bool_))
    ]
listaSelecciones2 = [1.0, 2.0]
dfNetflix["duration_type"] =  np.select(listaCondiciones, listaSelecciones, default='Not Specified')

# Variables 
netflixX = dfNetflix[["Cod_type","duration_type"]][:2000]                               
netflixY= dfNetflix["duracion"][:2000]   

# Estandarizacion 
scaleNetflix= StandardScaler()
scaleNetflix_X =scaleNetflix.fit_transform(netflixX)

#Train 
trainX= scaleNetflix_X[:1400]
trainY= netflixY[:1400]

#test
testX= scaleNetflix_X[1400:]
testY= netflixY[1400:]

#Modelo de regresion 
modelo = linear_model.LinearRegression()
modelo.fit(trainX,trainY)

#Prediccion 
pred_scale_x_netflix = modelo.predict([testX[0]])
print(pred_scale_x_netflix)

#R de relacion
r2_train = r2_score(trainY, modelo.predict(trainX))
print(r2_train)

r2_test = r2_score(testY, modelo.predict(testX))
print(r2_test)
