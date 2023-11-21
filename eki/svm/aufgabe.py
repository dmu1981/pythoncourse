import numpy as np
from matplotlib import pyplot as plt

# Gegeben seien die folgenden Daten 
N = 50
coords_red  = np.concatenate([
  np.array([[2,0],[0,1]]) @ np.random.normal(0, 1, (2,N)) + np.array([[0,3]]).T,
  np.array([[1,0],[0,2]]) @ np.random.normal(0, 1, (2,N)) + np.array([[0,-2]]).T,
],axis=1)

coords_blue  = np.concatenate([
  np.array([[1,-0.5],[0,2]]) @ np.random.normal(0, 1, (2,N)) + np.array([[-3,0]]).T,
  np.array([[1,0.5],[0,2]]) @ np.random.normal(0, 1, (2,N)) + np.array([[3,0]]).T
], axis=1)

X = np.concatenate([coords_red, coords_blue], axis=1).T
Y = np.concatenate([np.ones(coords_red.shape[1]), -np.ones(coords_blue.shape[1])]).T

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_normalized = scaler.transform(X)

# Aufgabe 1
# Importieren Sie das SkiLearn-Paket und trainieren sie eine Support Vektor Maschien auf den Daten
# 
#     https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#
# # Verwenden Sie einen Polynomial-Kernel Grad 2
def Aufgabe1():
  from sklearn import svm
  svc = svm.SVC(kernel="poly", degree=2)
  svc.fit(X_normalized, Y)
  return svc

# Aufgabe 2
# Erstellen Sie eine Prädiktion für jeden Punkt und zeichnen Sie korrekt klassifizierte 
# Punkte als grüne Diamanten sowie falsch klassifizierte Punkte als schwarze Kreuze.
# Hinweis: Grüne Diamante bekommen Sie mit dem "gD" Parameter, Schwarze Kreuze mit dem "kx" Parameter.
def Aufgabe2(svc):
  prediction = svc.predict(X_normalized)
  correctIndices = (prediction == Y)
  
  plt.plot(X[correctIndices, 0],X[correctIndices, 1], "gD")

  wrongIndices = (prediction != Y)
  plt.plot(X[wrongIndices, 0],X[wrongIndices, 1], "kx")

# Aufgabe 3
# Wieviel Prozent der Daten werden korrekt klassifiziert?
def Aufgabe3(svc):
  prediction = svc.predict(X_normalized)
  correctIndices = (prediction == Y)
  accuracy = np.sum(correctIndices) / Y.shape[0]
  print(f"Accuracy={100.0*accuracy:.3f}%")

# Aufgabe 4
# Experimentieren Sie mit verschiedenen Kerneln und Parametrisierungen um die Genauigkeit zu erhöhen
def Aufgabe4():
  # Hier ist nichts zu tun, verstellen Sie den Kernel und parametrisieren Sie diesen ggf. anders
  pass

# Aufgabe 5
# Implementieren sie eine Hyperparametersuche für die besten Hyperparameter GAMMA und C des RBF-Kernels.
# Durchsuchen Sie dazu mit np.linspace(0, 2.5, 20) und np.geomspace(0.1, 1000, 20) verschiedenen Werte für GAMMA und C,
# fitten sie jeweils die SVM und bestimmen Sie die Genauigkeit. Behalten Sie die Parameter, welche die beste Genauigkeit liefern
def Aufgabe5():
  from sklearn import svm
  best_gamma = 0
  best_C = 0
  best_accuracy = 0
  for gamma in np.linspace(1.0, 5.0, 30):
    for C in np.geomspace(0.1, 1000, 20):
      svc = svm.SVC(kernel="rbf", gamma=gamma, C=C)
      svc.fit(X_normalized, Y)
      prediction = svc.predict(X_normalized)
      correctIndices = (prediction == Y)
      accuracy = np.sum(correctIndices) / Y.shape[0]
      if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_gamma = gamma
        best_C = C
        
        print(f"Neuer Bestwert: gamma={gamma:.3f}, C={C:.3f}, accuracy={100*accuracy:.3f}%")
  
  best_svc = svm.SVC(kernel="rbf", gamma=best_gamma, C=best_C)
  best_svc.fit(X_normalized, Y)
  return best_svc

# Aufgabe 6
# Zeichnen Sie zu ihrer besten SVM aus Aufgabe 5 die Trennfläche als Contour. 
# Erzeugen Sie dazu mit np.linspace(-8,8,50) zwei lineare
# Bereichen von jeweils -8 bis 8 für die X und Y Werte. 
# Benutzen Sie den scaler von oben um jeden dieser Punkte zu normalisieren und prädizieren Sie
# den Punkt mit der SVM. Speichern Sie alle Prädiktionen in einem Array (Hinweis: Sie können mit z=np.zeros((50,50)) starten)
# und erstellen Sie dann einen Contour-Plot mit 
# 
#     plt.contourf(xx, yy, z, levels=[-1,0,1], cmap="seismic", alpha=0.2)

def Aufgabe6(svc):
  N = 150
  x_range = np.linspace(-8,8,N)
  y_range = np.linspace(-8,8,N)

  z = np.zeros((N, N))
  for x_index, x  in enumerate(x_range):
    for y_index, y in enumerate(y_range):
      coord = scaler.transform(np.array([[x,y]]))
      z[y_index, x_index] = svc.predict(coord)

  xx, yy = np.meshgrid(x_range, y_range)
  plt.contourf(xx, yy, z, levels=[-1,0,1], cmap="seismic", alpha=0.2)



if __name__ == "__main__":
  svc = Aufgabe1()
  #Aufgabe2(svc)
  #Aufgabe3(svc)
  best_svc = Aufgabe5()
  Aufgabe6(best_svc)

  plt.plot(coords_red[0], coords_red[1], "r.")  
  plt.plot(coords_blue[0], coords_blue[1], "b.")
  plt.show()
