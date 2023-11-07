import numpy as np
from cars import load_cars_data, Features

# Load cars data
train,test  = load_cars_data(True)

# Take features from test and training set
x_train = train.take((Features.Bias, Features.Year, Features.KM, Features.Fuel, Features.Seller, Features.Transmission), axis=1)
y_train = train.take((Features.Price), axis=1)

x_test = test.take((Features.Bias, Features.Year, Features.KM, Features.Fuel, Features.Seller, Features.Transmission), axis=1)
y_test = test.take((Features.Price), axis=1)

# Aufgabe 1
# Bestimmen Sie ein lineares Regressionsmodell für den Verkaufspreis
# Verwenden Sie dazu die Pseudoinversen wie in der Vorlesung gezeigt
# Regularisieren Sie ihr Modell leicht (z.B. mit 0.01*np.eye(6))
# Diskutieren Sie die Parameter ihres Modells in Hinblick auf den untersuchten Sachzusammenhang
# (z.B. welches Vorzeichen hat der Koeffizient für das "Kilometer Driven" Feature und was bedeutet dies für die Modellvorhersage?)



# Aufgabe 2
# Bestimmen Sie den mittleren quadratischen Fehler ihres Vorhersagemodells auf dem Trainigs und Test-Set
# Wie interpretieren Sie die Differenz der Werte?



# Aufgabe 3
# Welchen Händler-Verkaufspreis visieren Sie an für einen 2013 gebauten 
# Benziner mit 9800 Kilometer Fahrleistung und Automatikgetriebe?


