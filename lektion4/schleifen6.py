staedte = [
    ("Schleswig-Holstein","Kiel"),
    ("Hamburg","Hamburg"),
    ("Mecklenburg-Vorpommern","Schwerin"),
    ("Niedersachsen","Hannover"),
    ("Bremen","Bremen"),
    ("Sachsen-Anhalt","Magdeburg"),
    ("Brandenburg","Potsdam"),
    ("Berlin","Berlin"),
    ("Sachsen","Dresden"),
    ("Thüringen","Erfurt"),
    ("Hessen","Wiesbaden"),
    ("Nordrhein-Westfalen","Düsseldorf"),
    ("Rheinland-Pfalz","Mainz"),
    ("Saarland","Saarbrücken"),
    ("Baden-Württemberg","Stuttgart"),
    ("Bayern","München")
]

for index, tupel in enumerate(staedte):
    bundesland, hauptstadt = tupel
    print(f"{index:2}: Die Hauptstadt von {bundesland} ist {hauptstadt}")
    if bundesland == hauptstadt:
        print(f"    {hauptstadt} ist ein Stadtstaat")