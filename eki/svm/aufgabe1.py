import numpy as np
from stars import load_stars

# Load cars data
train, test = load_stars(True)

# Take features from test and training set
x_train = train.take((0,1,2,3,4,5,6,7), axis=1)
y_train = train.take((8), axis=1)

x_test = test.take((0,1,2,3,4,5,6,7), axis=1)
y_test = test.take((8), axis=1)


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
# Importieren Sie die SVC (Support Vector Classification) von sklearn
#
#   https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#
# und trainieren Sie eine Support Vektor Maschine (C=1) mit linearem Kernel
from sklearn.svm import SVC
svc = SVC(kernel="linear", C=10)
svc.fit(x_train, y_train)

# Aufgabe 3
# Bestimmen Sie die Genauigkeit ihres Klassifikators auf dem Trainings sowie dem Testset 
pred = svc.predict(x_test)
c0 = (y_test == 0)
c1 = (y_test == 1)
c0acc = np.sum(pred[c0] == y_test[c0]) / np.sum(c0)
c1acc = np.sum(pred[c1] == y_test[c1]) / np.sum(c1)

correct_test = (c0acc + c1acc) / 2.0

pred = svc.predict(x_train)
c0 = (y_train == 0)
c1 = (y_train == 1)
c0acc = np.sum(pred[c0] == y_train[c0]) / np.sum(c0)
c1acc = np.sum(pred[c1] == y_train[c1]) / np.sum(c1)

correct_train = (c0acc + c1acc) / 2.0

print(f"Train: {correct_train*100.0:.2f}%, Test: {correct_test*100.0:.2f}%")
