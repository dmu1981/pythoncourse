import math

a = 2
b = 7
print(f"{a} hoch {b} = {a**b}")
print(f"{a} / {b} = {round(a / b, 1)} (auf 1 Stelle gerundet)")
print(f"{b} mod {2} = {b % a} (Rest bei ganzahliger Division)")

print(f"{a} / {b} = {math.ceil(a / b)} (aufrunden)")
print(f"{a} / {b} = {math.floor(a / b)} (abrunden)")
