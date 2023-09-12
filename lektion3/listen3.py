primzahlen = [2,3,5,7,11,13,17,19]
vorgaenger = [x-1 for x in primzahlen]

paare = list(list(zip(vorgaenger, primzahlen)))
print(paare)

