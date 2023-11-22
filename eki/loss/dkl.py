import numpy as np
from matplotlib import pyplot as plt

# Wir wollen in diesem Beispiel die theoretische Gleichverteilung eines Würfels gegenüber
# einer aus einer Stichprobe beobachteten Verteilung vergleiche. Dazu wollen wir die
# Kullbach-Leibler Divergenz zwischen beiden berechnen

# Die theoretische Verteilung p ist gegeben durch
p = { augenzahl: 1.0 / 6.0 for augenzahl in range(1,7) }
# Wir wollen 1000 Würfelwürfe durchführen
N = 2000

# Aufgabe 1
# Verwenden Sie np.random.uniform und np.ceil um N Würfelwürfe zu simulieren
dice = np.ceil(np.random.uniform(0, 6, N))
print("dice: ", dice)

# Aufgabe 2
# Zählen Sie nun wie oft sie die Augenzahlen 1 bis 6 getroffen haben und bestimmen Sie die
# empirische Massenfunktion q ähnlich wie oben p
q = {}
for augenzahl in range(1,7):
    q[augenzahl] = np.sum(dice==augenzahl) / N
print("q: ", q)

# Aufgabe 3
# Bestimmen Sie nun die Kullbach-Leibler Divergenz zwischen P und Q nach Skript
dkl = np.sum([p[augenzahl] * np.log(p[augenzahl] / q[augenzahl]) for augenzahl in range(1,7)])
print("dkl: ", dkl)

# Aufgabe 4
# Nutzen Sie plt.bar um die Massenfunktion Q darzustellen 
plt.bar(q.keys(), q.values())
plt.show()