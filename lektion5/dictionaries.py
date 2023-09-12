# Dictionaries speichern ungeordnete Key-Value Paare
student = {
    "name": "Twenty 4 Tim",
    "matrikelnummer": 392918,
    "studiengang": "Deutsch, Englisch auf Lehramt"
}

# Werte können anhand ihres keys identifiziert werden
print(student["name"])

# Neue Werte können ebenfalls hinzugefügt werden
student["geburtsdatum"] = "14.09.2000"

# Mit dem *in* Keyword kann überprüft werden ob es einen
# bestimmten Key gibt
print("name" in student)
print("geschlecht" in student)

# Die Funktion "items" liefert alle Key/Value Paare als Tupel
for key, value in student.items():
    print(f"{key:20}: {value}")