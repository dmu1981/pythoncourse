# Wir starten mit allen Zahlen zwischen 1 und 10 (inklusive)
zahlen = range(1,11)

# Wir bilden jede Zahl auf ihr Quadrat ab
quadratzahlen = map(lambda x: x**2, zahlen)

# Filtere die ungeraden heraus
gerade = filter(lambda x: x%2==0, quadratzahlen)

# Erzeuge eine Liste und gib diese aus
print(list(gerade))

    