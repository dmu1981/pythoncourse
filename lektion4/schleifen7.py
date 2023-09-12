# Das Sieb des Eratosthenes
# Wir beginnen mit allen Zahlen zwischen 2 und 1000
sieb = list(range(2,1000))

# Am Anfang kennen wir keine Primzahlen
primzahlen = []

# Solange noch Kandidaten übrig sind
while len(sieb) > 0:
    # Die erste Zahl in unserer Liste ist eine Primzahl
    # weil sie von keiner vorherigen Zahl geteilt wurde
    primzahl = sieb[0]
    primzahlen.append(primzahl)

    # "Siebe" nun alle Zahlen aus die von "primzahl" geteilt werden
    sieb = [x for x in sieb if x%primzahl!=0]    

print(primzahlen)