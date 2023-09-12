x = int(input("Bitte geben sie eine natürliche Zahl ein:"))
if x < 0:
    x = 0
    print('Negative Zahlen sind keine natürlichen Zahlen!')
elif x == 0:
    print('Null ist keine natürliche Zahl!')
elif x == 1:
    print('Eins ist die erste natürliche Zahl')
else:
    print('Puh, ganz schön viel')