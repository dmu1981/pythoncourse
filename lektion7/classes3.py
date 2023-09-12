class Professor:
    def __init__(self, name, fachgebiet):
        self.name = name
        self.fachgebiet = fachgebiet

class Pruefungsleistung:
    def __init__(self, titel, pruefer, note):
        self.titel = titel
        self.pruefer = pruefer
        self.note = note

    def bestanden(self):
        return self.note <= 4.0
    
    def print(self):
        raise NotImplemented()
    
class MuendlichePruefung(Pruefungsleistung):
    def __init__(self, titel, pruefer, note, beisitzer):
        super().__init__(titel, pruefer, note)
        self.beisitzer = beisitzer

    def print(self):
        print(f"Mündliche Prüfung: {self.titel} bei {self.pruefer.name}, Note: {self.note}. Beisitzer {self.beisitzer}")

class SchriftlichePruefung(Pruefungsleistung):
    def __init__(self, titel, pruefer, note, laenge):
        super().__init__(titel, pruefer, note)
        self.laenge = laenge

    def print(self):
        print(f"Schriftliche Prüfung: {self.titel} bei {self.pruefer.name}, Note: {self.note}. Länge {self.laenge} Minuten")

dennisMueller = Professor("Dennis Müller", "Künstliche Intelligenz")

pruefungen = []
pruefungen.append(MuendlichePruefung("Elektrotechnik II", dennisMueller, 1.0, "Jochen Feitsch"))
pruefungen.append(SchriftlichePruefung("Mathematik I", dennisMueller, 1.3, 120))

for pruefung in pruefungen:
    pruefung.print()