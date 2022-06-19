#Diplomado python 
#Autor: Jonnier Andres Teran Morales 
#ID No:502195
#Id:1003064599
#ID:correo:Jonnier.teran@upb.edu.co
#Cel:3245644212

#librerias 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import r2_score


scale = StandardScaler()

#Importar base de datos
dfStudent = pd.read_csv("student_data.csv")


# Variable dependiente / independiente 
x = dfStudent[["age","freetime"]]
y = dfStudent[["health"]]

#Estandarizacion de los datos 

scaledX = scale.fit_transform(x)

#Train- Test
trainX = scaledX[:300]
trainY = y[:300]

#Test
testX = scaledX[300:]
testY = y[300:]

#Modelo de regresion 
modelo = linear_model.LinearRegression()
modelo.fit(trainX,trainY)

#Prediccion del modelo
pred_scaleX = modelo.predict([testX[0]])
print("la Predicion es:")
print(pred_scaleX)

r2_train = r2_score(trainY, modelo.predict(trainX))
print(r2_train)

r2_test = r2_score(testY, modelo.predict(testX))
print(r2_test)



