# Wir brauchen zwei Bibliotheken
import math
import random

# Wir beginnen mit 10 Punkten
anzahlPunkte = 10

# Maximal eine Million Punkte
MAXIMALE_PUNKTE = 1000000

# Erzeuge einen zufälligen Punkt im Einheitsquadar
def zufaelliger_punkt():
  return (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))

# Bestimme den Abstand eines Punktes zum Ursprung
def abstand(punkt):
  return math.sqrt(punkt[0]**2 + punkt[1]**2)

# Wiederhole mehrfach solange die Anzahl Punkte noch nicht zu groß geworden ist
while anzahlPunkte < MAXIMALE_PUNKTE:
  # Bestimme zufällige Punkte im Einheitsquadrat
  quadrat = [zufaelliger_punkt() for _ in range(anzahlPunkte)]

  # Bestimme die Teilmenge der Punkte die auch im Einheitskreis liegen
  kreis = [x for x in quadrat if abstand(x) < 1]

  # Schätze Pi anhand dieses Verhältnisses
  pi = len(kreis) / anzahlPunkte * 4

  # Bestimme den relativen Fehler gegenüber dem korrekten Wert
  fehler = 100.0 * (pi / math.pi - 1)

  # Gib das ganze nett aus
  print(f"{anzahlPunkte:6} Punkte: π ≈ {pi:.6f} ({fehler:6.2f}% Fehler)")

  # Und verdopple die Anzahl Punkte für den nächsten Durchlauf
  anzahlPunkte *= 2