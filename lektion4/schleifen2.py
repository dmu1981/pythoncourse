# Zwei Listen definieren
greetings = ["Hallo", "Welcome", "Ni Hao"]
namen = ["Anton", "Beate", "Clara"]

# Iteriere über alle Begrüßungen
for greeting in greetings:
    # Iterieren über alle Namen
    for name in namen:
        # Gib eine entsprechende Begrüßung aus
        print(f"{greeting} {name}")