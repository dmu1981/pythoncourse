def berechne(x, y):
    return (x + y), (x - y), (x * y), (x / y)

a = 3
b = 4

summe, differenz, produkt, quotient = berechne(a, b)
print(f"Die Summe aus {a} und {b} ist {summe}")
print(f"Die Differenz aus {a} und {b} ist {differenz}")
print(f"Das Produkt aus {a} und {b} ist {produkt}")
print(f"Der Quotient von {a} und {b} ist {quotient}")