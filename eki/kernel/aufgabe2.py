import numpy as np
from cars import load_cars_data, Features

# Load cars data
train,test  = load_cars_data(True)

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
from sklearn import preprocessing
transformer = preprocessing.StandardScaler().fit(x_train)
x_train = transformer.transform(x_train)
x_test = transformer.transform(x_test)

# Aufgabe 2
# Importieren Sie sklearn und verwenden sie sklearn.kernel_ridge.KernelRidge
#
#   https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge
#
# um ein Regressionsmodell mit polynomialkernel vom Grad 2 zu berechnen. Verwenden Sie alpha=0.01 zur Regularisierung
# Verwenden Sie das Trainingsset
from sklearn.kernel_ridge import KernelRidge
krr = KernelRidge(alpha=0.01, kernel="poly", degree=2)
krr.fit(x_train, y_train)
 
# Aufgabe 3
# Berechen Sie wieder den mittleren quadratischen Fehler auf beiden Sets
train_error = np.sum((krr.predict(x_train) - y_train)**2) / x_train.shape[0]
test_error = np.sum((krr.predict(x_test) - y_test)**2) / x_test.shape[0]
print(train_error, test_error)

# Aufgabe 4
# Erstellen Sie erneut eine Prognose für einen 2013 gebauten 
# Benziner mit 9800 Kilometer Fahrleistung und Automatikgetriebe?
# Beachten Sie das Sie ihre Daten zunächst mit dem StandardScaler ebenfalls normalisieren müssen!
pred = transformer.transform(np.array([[2013, 9800, 0, 0, 1]]))
v = krr.predict(pred)
print(v)

