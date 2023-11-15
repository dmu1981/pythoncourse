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
# und trainieren Sie eine Support Vektor Maschine (C=1) mit polynomial kernel (Grad 2) 
from sklearn.svm import SVC
svc = SVC(kernel="poly", C=1, degree=2)
svc.fit(x_train, y_train)

# Aufgabe 3
# Bestimmen Sie die Genauigkeit ihres Klassifikators auf dem Trainings sowie dem Testset 
correct_train = np.sum(svc.predict(x_train) == y_train) / y_train.shape[0]
correct_test = np.sum(svc.predict(x_test) == y_test) / y_test.shape[0]
print(f"Train: {correct_train*100.0:.2f}%, Test: {correct_test*100.0:.2f}%")

# Aufgabe 4
# Bestimmen Sie nun das optimale C mit Hilfe einer Hyperparameter-Analyse. 
# Iterieren Sie dazu über verschiedene Wert von C (z.b. über np.geomspace(0.1,10,100), schätzen Sie jeweils eine
# Support Vektor Maschine, bestimmen Sie jeweils die Genauigkeit auf dem Testset und behalten Sie diejenige, die 
# eine möglichst große Testgenauigkeit erreicht
from tqdm import tqdm
bar = tqdm(np.geomspace(0.1, 10.0, 100))
best_C = None
best_acc = 0
for C in bar:
    svc = SVC(kernel="poly", C=C, degree=2)
    svc.fit(x_train, y_train)
    correct_test = np.sum(svc.predict(x_test) == y_test) / y_test.shape[0]
    if correct_test > best_acc:
        best_acc = correct_test
        best_C = C
        bar.set_description(f"acc: {best_acc*100.0:.2f}%, C={best_C:.5f}")

