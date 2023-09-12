# Wir suchen Primzahlen bis N = 100
N = 100

primzahlen = []

for kandidat in range(2,N):    
    for teiler in range(2, kandidat-1):
        if kandidat % teiler == 0:
            hat_teiler = True
            break
    else:    
        primzahlen.append(kandidat)

print(primzahlen)