import math
import numpy as np
from matplotlib import pyplot as plt

# In dieser Übung sollen sie den Perzeptron-Lernalgorithmus aus der Vorlesung implementieren.
# Dazu betrachten wir zunächst die beiden Mengen P und N 
P = np.array([[1.0,1.0,1.0],
              [2.0,2.0,1.0],
              [4.0,3.0,1.0],
              [1.0,3.5,1.0],
              [1.5,2.5,1.0]
              ])

N = np.array([[3.0,1.0,1.0],
              [4.0,2.0,1.0],
              [5.0,2.0,1.0],
              [4.0,1.0,1.0],
              [3.0,1.5,1.0]
              ])

# Umser Perzeptron verwendet die Entscheidungsfunktion 
#
#                    w * x >= 0     (1)
#   w1 * x1 + w2 * x2 + w3 >= 0     (2)
#
# wobei x ein Vektor aus P oder N ist und w = (w1, w2, w3) der dreidimensionale 
# Parametervektor des Perzeptrons.

# Aufgabe 1
# Bestimmen Sie ausgehend von dem Gewichtsvektor w = (-0.5, 1.0, 0.5) für jeden Punkt aus P und N, wie
# dieser von Perzeptron klassifiziert werden würde
w = np.array([-0.5, 1.0, 0.5])
print(P @ w > 0)
print(N @ w > 0)

# Aufgabe 2
# Implementieren Sie eine Funktion "plot_data" welche die beiden Mengen P und N sowie
# die durch w bestimmte Trennfläche plottet. Stellen Sie dazu Gleichung (2) (s. oben)
# nach y um. 
def plot_data(P, N, w):
    plt.plot(P[:,0], P[:,1], 'bs')
    plt.plot(N[:,0], N[:,1], 'ro')

    x_range = np.linspace(0,6,20)
    y = -w[2]/w[1] - w[0]/w[1]*x_range
    plt.plot(x_range, y, "k")

# Aufgabe 3
# Konkatenieren Sie nun wie in der Vorlesung die Menge P mit der negierte Menge N um, wie in der Vorlesung, eine Menge 
# B zu erhalten in der alle Datenpunkte x € B auf der positiven Seite der Trennebene liegen müssen.
# Wie überprüfen Sie ob alle Punkte gleichzeitig korrekt klassifiziert werden?
B = np.concatenate([P, -N],0)
print(np.sum(B @ w > 0) == B.shape[0])

# Aufgabe 4
# Implementieren Sie nun den Perzeptron-Lernalgorithmus wie in der Vorlesung gezeigt.
# Starten Sie mit einer While-Schleife, die solange ausgeführt wie nicht alle Punkte
# korrekt klassifiziert werden. Wählen sie mit np.random.uniform einen zufälligen
# Datenpunkt aus B. Überprüfen Sie ob dieser falsch klassifiziert wird (x) und passen sie ggf. 
# den Gewichtsvektor gemäß Vorlesung an. Nutzen Sie die plot_data Methode von oben um den neuen 
# Zustand zu visualisieren. 
#
# Hinweis: Sie müssen mit plt.ion() den interaktiven Modus von matplotlib aktivieren. 
# Mit plt.clf() können sie die aktuelle Figure löschen (zurücksetzen) und mit plt.pause(1) können sie
# eine Sekunde warten (und in dieser Zeit die Grafik anzeigen) bevor das Programm weiter macht.
plt.ion()
while np.sum(B @ w > 0) < B.shape[0]:
    # Pick a random vector
    index = math.floor(np.random.uniform(0, B.shape[0]))
    
    # Get the data point
    x = B[index, :]

    # If it is correctly classified, skip
    if w @ x > 0.0:
        print(w @ x)
        continue

    # Update the W Vektor
    w = w + x

    # Draw everything
    plt.clf()
    plot_data(P, N, w)
    plt.xlim(0,6)
    plt.ylim(0,4)
    plt.pause(0.1)

print("Done")
plt.pause(5.0)    

