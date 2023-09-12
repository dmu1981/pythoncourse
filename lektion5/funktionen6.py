def highlight(x):
    print(f" --> {x} <--")

def normal(x):
    print(x)

def ausgabe(formater, n=10):
  for x in range(n):
     formater(x)

ausgabe(normal)
ausgabe(highlight, n=5)
    
    