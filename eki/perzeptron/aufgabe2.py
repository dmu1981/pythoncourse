import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

# Wir wollen nun ein Perzeptron auf dem Handwritten Digits Datenset von SKLEARN trainieren.

# Aufgabe 1
# Laden Sie das Datenset, extrahieren sie die Daten und die Zielklassen.
#
#   https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset
#
data = datasets.load_digits()
X = data["data"]
Y = data["target"]

# Aufgabe 2
# Erzeugen Sie mit 
# 
#   fig, axs = plt.subplots(5,8) 
# 
# eine Menge von insgesamt 40 plots. Wählen Sie dann für jeden dieser Plots ein zufälliges 
# Bild und verwenden sie 
# 
#   X[index,:].reshape(8,8) 
# 
# um den Vektor in ein Bild umzuformen. Verwenden Sie 
# 
#   axs[row,col].imshow(...)
#
# um dieses Bild anzuzeigen. Wählen Sie eine geeignete Colormap über die cmap Option
# und setzen sie den Titel des Plots auf die jeweilige Klasse 
fig, axs = plt.subplots(5,8)
for row in range(5):
  for col in range(8):
    index = int(np.floor(np.random.uniform(0.0, X.shape[0]-1)))

    axs[row,col].imshow(X[index,:].reshape(8,8), cmap="gray")
    axs[row,col].set_title(Y[index])
plt.show()

# Aufgabe 3
# Unser Perzeptron soll die Klasse 4 von allen anderen Klassen unterscheiden
# Finden sie dazu alle Indices der positiven (4) und negativen (nicht 4) Klasse.
# Invertieren Sie wie im Skript die Datenpunkte der negativen Klasse (vgl. vorherige Aufgabe)
positive_indices = (Y == 4)
negative_indices = (Y != 4)
X[negative_indices] *= -1

# Aufgabe 4
# Wählen sie mit np.random.normal(...) einen normalverteilten Startvektor w der richten Größe
# Passen Sie dann den Perzeptron-Lernalgorithmus aus der vorherigen Aufgabe für dieses Problem an.
# ACHTUNG:
# Wir wissen nicht sicher ob die Daten linear trennbar sind, d.h. wir wissen nicht ob der Algorithmus
# konvergiert. Ändern Sie ihn entsprechend indem sie konstant 1000 Iterationen durchführen anstatt 
# so lange zu warten bis alle Punkte korrekt klassifiziert werden. 
# Bestimmen sie nach jedem Update die Genauigkeit (Accuracy) ihres Perzeptrons und geben Sie diese auf der
# Konsole aus. 
w = np.random.normal(0.0, 1.0, 64)

for step in range(1000):
  index = int(np.floor(np.random.uniform(0.0, X.shape[0]-1)))
  if X[index, :] @ w > 0:
    continue
  
  w = w + X[index,:]

  correct = np.sum(X @ w > 0)
  acc = correct / X.shape[0]
  print(f"Accuracy: {100.0*acc:.2f}%")

