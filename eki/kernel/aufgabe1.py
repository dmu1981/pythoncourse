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
model = np.linalg.inv(x_train.T @ x_train + 0.01*np.eye(6)) @ x_train.T @ y_train
print(model)
"""
    Der "YEAR"-Koeffizient ist positiv, d.h. jüngere Autos erzielen einen höheren Verkaufspreis
    Der "KM"-Koeffizient ist negativ, d.h. mehr Fahrleistung reduziert den Verkaufspreis
    Der "Fuel"-Koeffizient ist positiv, d.h. Dieselfahrzeuge erzielen einen höheren Verkaufspreis
    Der "Seller"-Koeffizient ist negativ, d.h. Privatverkäufe erzielen einen niedrigeren Verkaufspreis
    Der "Transmission"-Koeffizient ist positiv, d.h. Automatikwagen erzielen einen höheren Verkaufspreis
"""

# Aufgabe 2
# Bestimmen Sie den mittleren quadratischen Fehler ihres Vorhersagemodells auf dem Trainigs und Test-Set
# Wie interpretieren Sie die Differenz der Werte?
train_error = (1/x_train.shape[0])*np.sum((x_train @ model - y_train)**2)
test_error = (1/x_test.shape[0])*np.sum((x_test @ model - y_test)**2)
print(train_error, test_error)

# Aufgabe 3
# Welchen Händler-Verkaufspreis visieren Sie an für einen 2013 gebauten 
# Benziner mit 9800 Kilometer Fahrleistung und Automatikgetriebe?
prediction = model.T @ np.array([1, 2013, 9800, 0, 0, 1])
print(prediction)