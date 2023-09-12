# Eine Liste mit Namen erzeugen
namen = ["Anton", "Beate", "Clara", "Denise", "Edin"]
print(namen)

# Mit dem "in" Operator kann überprüft werden
# ob ein Element in der Liste vorkommt oder nicht
print("Anton" in namen)
print("Felix" in namen)

# Elemente können mehrfach in der Liste sein 
# (Listen sind keine Mengen)
namen = namen + ["Anton", "Beate"]
print(namen)

# Elemente in Listen können direkt indiziert werden. 
# Man beachte das das erste Element den Index 0 trägt
print(f"Der zweite Name ist {namen[1]}")

# Mit len(...) kann die Anzahl Elemente in einer Liste
# bestimmt werden
print(f"Es gibt {len(namen)} Namen")

# Python kennt auch echte Mengen (set)
namen = set(namen)
print(namen)
print(f"Es gibt {len(namen)} eindeutige Namen")
