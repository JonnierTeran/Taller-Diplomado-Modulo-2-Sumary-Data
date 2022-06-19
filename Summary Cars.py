#Diplomado python 
#Autor: Jonnier Andres Teran Morales 
#ID No:502195
#Id:1003064599
#ID:correo:Jonnier.teran@upb.edu.co
#Cel:3245644212

#importamos librerias 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


#importamos la base de datos 
dfCars = pd.read_csv("cars2.csv")

#variables independiente y dependiente 
x = dfCars[["Weight"]]
y = dfCars[["CO2"]]


#Estandarizacion  de los datos 
scale = StandardScaler()

scaledX = scale.fit_transform(x)


#train / test
trainX = scaledX[:25]
trainY = y[:25]

#test
testX = scaledX[25:]
testY = y[25:]

# modelo 
modelo = linear_model.LinearRegression().fit(trainX, trainY)
print(modelo.score(trainX,trainY))

print(modelo.score(testX, testY))

#Prediccion del modeloo
pred_scaledX = modelo.predict([testX[0]])
print("Predicion es:")
print(pred_scaledX)


