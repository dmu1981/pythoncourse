def quotient(divident, divisor):
    if divisor == 0:
        raise Exception("Durch 0 kann man nicht teilen!")
    
    return divident / divisor

print(quotient(6, 3))

try:
    print(quotient(6, 0))
except Exception as e:
    print("Ups, hier kam es zu einer Außnahme!")

print(quotient(6, 1))    

# Diese Außnahme wird nicht abgefangen und beendet das Program!
print(quotient(6, 0)) 

# Dies wird niemals ausgeführt werden!
print("Hallo Welt") 