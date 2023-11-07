import numpy as np
from cars import load_cars_data, Features

# Load cars data
train, test = load_cars_data(True)

# Take features from test and training set
x_train = train.take((Features.Year, Features.KM, Features.Fuel, Features.Seller, Features.Transmission), axis=1)
y_train = train.take((Features.Price), axis=1)

x_test = test.take((Features.Year, Features.KM, Features.Fuel, Features.Seller, Features.Transmission), axis=1)
y_test = test.take((Features.Price), axis=1)

# Aufgabe 1
# Importieren Sie sklearn und verwenden sie sklearn.preprocessing.StandarScaler 
#
#   https://scikit-learn.org/stable/modules/preprocessing.html
# 
# um die Daten zu normalisieren. 




# Aufgabe 2
# Schätzen Sie nun ein Modell mit RBF Kernel (gamma=0.3) mit dem Trainingsset (wieder alpha=0.01). 
# Welchen Fehler messen Sie auf dem Trainings und Test Set?




# Aufgabe 3
# Erstellen Sie erneut eine Prognose für einen 2013 gebauten 
# Benziner mit 9800 Kilometer Fahrleistung und Automatikgetriebe?
# Beachten Sie das Sie ihre Daten zunächst mit dem StandardScaler ebenfalls normalisieren müssen!



