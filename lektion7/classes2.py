class Pruefungsleistung:
    def __init__(self, titel, pruefer, note):
        self.titel = titel
        self.pruefer = pruefer
        self.note = note

    def bestanden(self):
        return self.note <= 4.0

class Professor:
    def __init__(self, name, fachgebiet):
        self.name = name
        self.fachgebiet = fachgebiet

class Student:
    def __init__(self, name, matrikelnummmer):
        self.name = name
        self.matrikelnummer = matrikelnummmer
        self.pruefungen = []

    def durchschnittsnote(self):
        gesamtnote = 0
        for pruefung in self.pruefungen:
            gesamtnote = gesamtnote + pruefung.note

        return gesamtnote / len(self.pruefungen)

    def pruefen(self, leistung):
        self.pruefungen.append(leistung)

class Universitaet:
    def __init__(self, name):
        self.name = name
        self.studenten = []
        self.professoren = []

    def berufen(self, prof):
        self.professoren.append(prof)

    def immatrikulieren(self, student):
        for st in self.studenten:
            if st.matrikelnummer == student.matrikelnummer:
                raise Exception("Matrikelnummer dürfen nicht doppelt vergeben werden")
            
        self.studenten.append(student)

    def getStudent(self, name) -> Student:
        for student in self.studenten:
            if student.name == name:
                return student
            
        return None
    
    def getStudenten(self):
        return self.studenten
    
    def getProfessor(self, name):
        for professor in self.professoren:
            if professor.name == name:
                return professor
            
        return None
    
    def printProfessoren(self):
        for prof in self.professoren:
            print(f"{prof.name} unterrichtet {prof.fachgebiet}")

        

hsd = Universitaet("Hochschule Duesseldorf")
hsd.berufen(Professor("Dennis Müller", "Künstliche Intelligenz"))
hsd.berufen(Professor("Florian Huber", "Data Science"))
hsd.immatrikulieren(Student("Max Mustermann", 313))
hsd.immatrikulieren(Student("Beate Musterfrau", 123))

student = hsd.getStudent("Max Mustermann")
student.pruefen(Pruefungsleistung("Mathematik 1", hsd.getProfessor("Dennis Müller"), 1.7))
student.pruefen(Pruefungsleistung("Data Science", hsd.getProfessor("Florian Huber"), 5.0))

student = hsd.getStudent("Beate Musterfrau")
student.pruefen(Pruefungsleistung("Mathematik 1", hsd.getProfessor("Dennis Müller"), 3.2))
student.pruefen(Pruefungsleistung("Data Science", hsd.getProfessor("Florian Huber"), 1.0))

for student in hsd.studenten:
  print(student.name, student.matrikelnummer)
  for pruefung in student.pruefungen:
    print(f"Titel: {pruefung.titel} bei {pruefung.pruefer.name}, Bestanden: {pruefung.bestanden()}")

  print(f"Durchschnittsnote: {student.durchschnittsnote()}")
