def kaffee(tassen=4, milch=True, zucker=False):
    s = f"Ich koche {tassen} Tassen Kaffee"
    if milch:
        s += " mit Milch"
    else:
        s += " ohne Milch"
    
    s += " und"

    if zucker:
        s += " mit Zucker"
    else:
        s += " ohne Zucker"

    s += "."
    
    print(s)

kaffee()
kaffee(8)
kaffee(milch=False, zucker=True)