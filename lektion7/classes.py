class Professor:
    def __init__(self, name, fachgebiet):
        self.name = name
        self.fachgebiet = fachgebiet

class Universitaet:
    def __init__(self, name):
        self.name = name
        self.professoren = []

    def berufen(self, prof):
        self.professoren.append(prof)

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
hsd.printProfessoren()
