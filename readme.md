# Python Course
## Python installieren
Installieren sie eine möglichst aktuelle Version von Python auf ihrem System. Laden Sie dazu [hier](https://www.python.org/downloads/) den richtigen Installer für ihr System herunter und führen Sie diesen aus. Alternativ können Sie Python auch über den [Microsoft Store](https://learn.microsoft.com/de-de/windows/python/beginners) installieren

Überprüfen sie die korrekte Installation von Python auf ihrem System indem sie

    > python --version
    Python 3.8.12

ausführen. Der Python-Intepreter zeigt Ihnen die aktuell installierte Version an. Stellen Sie sicher das sie mindestens Python 3.8 installiert haben (Hinweis: Die zur Zeit aktuelle Version ist 3.11)

# Lektion 1 - Grundlagen
Alle folgenden Programme finden Sie in dem Ordner **lektion1**

## Hello World - 1
Starten Sie das "Hello World" (**helloworld1.py**) Program aus Lektion 1, indem sie

    > python lektion1/helloworld.py
    Hello World

ausführen. Das Programm besteht aus nur zwei Zeilen

    # Das ist ein Kommentar
    print("Hello World") 

Ein Hashtag (#) markiert den Beginn eines Kommentar. Python ignoriert
alle Zeichen nach dem Hashtag. Die Funktion "print" kann verwendet werden um etwas auf dem Terminal auszugeben. Das Argument der Funktion steht zwischen den Klammern, hier als "Hello World". Die Anführungszeichen markieren den Beginn und das Ende eines so genannten *strings*m, also einer Zeichenkette.

## Hello World - 2
Im Programm **helloworld2.py** finden Sie einen s.g. Multi-Line String.
Multi-Line Strings können über mehrere Zeilen gehen und werden mit drei aufeinanderfolgenden Anführungszeichen """ eingeleitet und beedent.

    print(
    """Hallo Welt!
    Dies ist ein Multi-Line String der über mehrere Zeilen gehen kann.

    Vielen Dank für die Aufmerksamkeit""")

## Hello World - 3
Das Programm **helloworld3.py** zeigt die Verwendung von Variablen.

    name="Welt"
    print(f"Hallo {name}")

Variablen speichern Zustände des Programms. In diesem Fall wird eine Variable mit den Namen "name" deklariert und ihr der Wert "Welt" zugewiesen, also die Zeichenkette bestehend aus den Buchstaben W, e, l und t. 

In der zweiten Zeile wird eine weitere Zeichenkette konstruiert. Das führende **f** zeigt an das es eine so genannte *formatierte* Zeichenkette sein soll. Bei diesen erlaubt ein spezieller Syntax das wir den Wert von Variablen in die Zeichenkette einsetzen. Die zu verwendende Variable wird dabei in geschweifte Klammern gesetzt. In diesem Fall wird also der Wert von "name", also "Welt" eingesetzt und der so entstehende String wird ausgegeben. 

## Hello World - 4
Strings können konkateniert (also hintereinander gehängt) werden. Dazu kann einfach der Additionsoperator + verwendet werden.

    hello="Hallo"
    world="Welt"
    print(hello + " " + world + "!")

# Lektion 2 - Arithmetik
Alle folgenden Programme finden Sie in dem Ordner **lektion2**

## Arithmetik
Das Programm **artithmetik.py** zeigt grundlegende arithmetische Operationen mit Variablen

    a = 20
    b = 5
    print(f"{a} + {b} = {a+b}")
    print(f"{a} - {b} = {a-b}")
    print(f"{a} * {b} = {a*b}")
    print(f"{a} / {b} = {a/b}")

Es erzeugt die folgende Ausgabe

    > python lektion2/artithmetik.py
    20 + 5 = 25
    20 - 5 = 15
    20 * 5 = 100
    20 / 5 = 4.0

Frage: Warum steht bei der letzten Zeilen 4.0 während bei den drei Ergebnissen davor die .0 fehlt?

## Arithmetik 2
Der Programm **artithmetik2.py** zeigt weitere arithmetische Operationen.

    import math

    a = 2
    b = 7
    print(f"{a} hoch {b} = {a**b}")
    print(f"{a} / {b} = {round(a / b, 1)} (auf 1 Stelle gerundet)")
    print(f"{b} mod {2} = {b % a} (Rest bei ganzahliger Division)")

    print(f"{a} / {b} = {math.ceil(a / b)} (aufrunden)")
    print(f"{a} / {b} = {math.floor(a / b)} (abrunden)")

Beachten Sie sie am Anfang mit 

    import math

ein zusätzliches Paket importiert wird welches zusätzliche Funktionen (in diesem Fall "ceil" und "floor") deklariert. Das Programm erzeugt
die folgende Ausgabe

    2 hoch 7 = 128
    7 mod 2 = 1 (Rest bei ganzahliger Division)
    2 / 7 = 1 (aufrunden)
    2 / 7 = 0 (abrunden)
    2 / 7 = 0.3 (auf 1 Stelle gerundet)

# Lektion 3 - Listen
Alle folgenden Programme finden Sie in dem Ordner **lektion3**

## Listen
Neben Strings, Ganzzahlen (Integers) und Fießkommazahlen (Floats) kennt Python auch so genannte Listen. Eine Liste ist dabei eine geordnete Auflistung (keine Menge!) von Elementen. Diese Elemenet müssen nicht unbedingt den selben Datentyp haben, sie können also Integers und Strings mischen, wenn Sie das wollen. 

    primzahlen = [2,3,5,7,11,13,17,19]
    print(primzahlen)

# Listen 2 - Konkatenation
Listen können aneinander gehängt werden durch den + Operator.
Beachten Sie das hier eben gerade nicht einzelnen Elemente miteinader addiert werden
    
    kleine_primzahlen = [2,3,5,7]
    grosse_primzahlen = [11,13,17,19]

    primzahlen = kleine_primzahlen + grosse_primzahlen
    print(primzahlen)

# Listen 3 - List Comprehension - Zip
Listen können implizit über andere Eigenschaften definiert werden. In diesem Beispiel

    primzahlen = [2,3,5,7,11,13,17,19]
    vorgaenger = [x-1 for x in primzahlen]

besteht die Liste "vorgaenger" aus den jeweiligen Vorgängern der Elemente in der Liste "primzahlen", also 

    vorgaenger = [1,2,4,6,10,12,16,18]

Mit dem "zip" Operator können zwei Listen wie ein Reißverschluß miteinander zu so genannten **Tupeln** verbunden werden. Dabei wird das erste Element der ersten Liste mit dem ersten Element der zweiten Liste verknüft, und so weiter und so fort. 

Der "list" Operator erzeugt daraus wieder eine Liste (das ist ein technisches Detail welches wir später diskutieren werden). Die so entstehende Liste kann mittels

    paare = list(list(zip(vorgaenger, primzahlen)))
    print(paare)

ausgegeben werden und erzeugt dann die folgende Ausgabe

    [(1, 2), (2, 3), (4, 5), (6, 7), (10, 11), (12, 13), (16, 17), (18, 19)]

# Listen 4 - List Comprehension 2
Bei der s.g. *list comprehension* kann über **if** eine zusätzliche Bedingung angegeben werden. 

    zahlen=[1,2,3,4,5,6,7,8,9,10]
    divisor=5
    teilbar=[x for x in zahlen if x%divisor == 0]
    print(teilbar)


    [5, 10]

# Listen 5 - Mengen vs. Listen 
Mit "in" kann überprüft werden ob ein Element in einer Liste vorkommt oder nicht

    namen = ["Anton", "Beate", "Clara", "Denise", "Edin"]
    print(namen)
    print("Anton" in namen)
    print("Felix" in namen)


    ["Anton", "Beate", "Clara", "Denise", "Edin"]
    True
    False

Dabei sind **True** und **False** s.g. Booleans, also binäre Wahrheitswerte.

Elemente können mehrfach in der Liste sein (Listen sind keine Mengen)

    namen = namen + ["Anton", "Beate"]
    print(namen)


    ['Anton', 'Beate', 'Clara', 'Denise', 'Edin', 'Anton', 'Beate']


Elemente in Listen können direkt indiziert werden. 
Man beachte das das erste Element den Index 0 trägt


    print(f"Der zweite Name ist {namen[1]}")


    Beate


Mit len(...) kann die Anzahl Elemente in einer Liste bestimmt werden

    print(f"Es gibt {len(namen)} Namen")
    
    
    Es gibt 7 Namen

Python kennt auch echte Mengen (set)

    namen = set(namen)
    print(namen)
    print(f"Es gibt {len(namen)} eindeutige Namen")


    {'Edin', 'Beate', 'Anton', 'Denise', 'Clara'}
    Es gibt 5 eindeutige Namen

# Listen 6 - Indices
Teile von Listen können über den ":" Operator indiziert werden.
Negative Indices bezeichnen dabei die hinteren Element der Liste.

    fibonnaci = [1,1,2,3,5,8,13,21,34]
    print(fibonnaci)

    # Gib die erste vier Elemente aus
    print(fibonnaci[:4])

    # Gib die letzten vier Elemente aus
    print(fibonnaci[-4:])

    # Auch Strings können indiziert werden
    text = "Donaudampfschiff"
    print(text[5:10])



# Aufgabe 1
* Erzeugen Sie eine Liste "doppelte", die aus den jeweils verdoppelten Primzahlen besteht. 

* Erzeugen Sie eine Variable "divident" und setzen Sie diese zunächst auf den Wert 71. Erzeugen Sie dann eine Liste "reste", die aus dem ganzzahligen Rest bei der Division von "divident" mit den ersten 8 Primzahlen entsteht. 

* Erzeugen Sie aus den Zahlen von 1 bis 10 eine Liste der jeweiligen  Quadratzahlen, sofern die Basis gerade (also durch 2 teilbar ist)

# Lektion 4 - Ablaufkontrolle
Alle folgenden Programme finden Sie in dem Ordner **lektion4**

## Bedingungen
Mit dem **if** Statement kann eine Bedingung überprüft werden und, in Abhängigkeit von der Bedingung, unterschiedlicher Programmcode ausgeführt werden. Die Bedingung muß dabei zu einem boolschen Wert evaluiert werden können (eine Wahrheitsaussage sein). Mit dem optionalen **else** kann festgelegt werden was passieren soll falls die Bedingung nicht erfüllt ist. 
Der konditionale Teil ist entsprechend "rechts eingerückt". Dadurch versteht Python welcher Codeblock zum "If-Zweig" und welcher zum "Else-Zweig" gehört. 

    a = 3
    b = 4

    if a > b:
        print("a ist größer als b")
    else:
        print("b ist größer als a")

## Bedingungen 2 - elif
Das **elif**-Statement steht für eine Abkürzung aus **else** und **if** und erlaubt die Verschachtelung mehrerer **else**-Zweige

    x = int(input("Bitte geben sie eine natürliche Zahl ein:"))
    if x < 0:
        x = 0
        print('Negative Zahlen sind keine natürlichen Zahlen!')
    elif x == 0:
        print('Null ist keine natürliche Zahl!')
    elif x == 1:
        print('Eins ist die erste natürliche Zahl')
    else:
        print('Puh, ganz schön viel')

## Schleifen
Mit dem **for** Statement können alle Elemente einer Liste iteriert werden

    namen = ["Anton", "Beate", "Clara", "Denise", "Edin"]
    for name in namen:
      print(f"Hallo {name}")

    print("Bis bald!")

dabei wird für jeden Durchlauf der Variable "name" durch das jeweilige Element der Liste ersetzt und dann der Schleifenkörper ausgeführt. Beachte das der Schleifenkörper durch entsprechendes "Rechts einrücken" vom Hauptprogramm abgetrennt ist. Damit weiß Python welcher Teil des Programms zum Schleifenkörper gehört und welcher nicht

## Schleifen 2 - Verschachtelung
Schleifen können verschachtelt werden um über mehrere Listen gleichzeitig zu iterieren.

    greetings = ["Hallo", "Welcome", "Ni Hao"]
    namen = ["Anton", "Beate", "Clara"]

    for greeting in greetings:
        for name in namen:
            print(f"{greeting} {name}")

## Schleifen 3 - continue
Die Elemente in Tupeln können ähnlich wie Listen direkt über einen 
Index angesprochen werden. Beachte das **zip** die beiden Listen "namen" und "alter" zu einem Tupel verknüpft. Die **for**-Schleife iteriert dann über alle (drei) Tupel. 

Der Befehl **continue** führt dazu das der aktuelle Schleifendurchlauf sofort abgebrochen wird und mit dem nächsten Element weitergemacht wird. 

    FSK = 18

    namen = ["Anton", "Beate", "Clara"]
    alter = [17, 19, 22]

    for tuple in zip(namen, alter):
        if tuple[1] < FSK:
            continue
        
        print(f"{tuple[0]} ist {tuple[1]} Jahre alt")

## Schleifen 4 - break
Der Befehl "range(start, stop)" liefert eine Liste von Integern zwischen "start" und "stop". [] entspricht der leeren Liste, also einer Liste ohne Elemente. Mittels "append" kann ein Element hinten an eine Liste angefügt werden. "break" unterbricht den Durchlauf der Schleife und setzt das Programm nach der Schleife fort. 

    N = 100

    primzahlen = []

    for kandidat in range(2,N):        
        hat_teiler = False
        for teiler in range(2, kandidat-1):
            if kandidat % teiler == 0:
                hat_teiler = True
                break
            
        if not hat_teiler:
            primzahlen.append(kandidat)

    print(primzahlen)

## Schleifen 5 - for else
Auch eine **for**-Schleife kann einen **else**-Zweig besitzen. Dieser wird nur dann ausgeführt wenn die Schleife nicht vorher durch ein **break** abgebrochen wurde. Damit läßt sich der Primzahlcode vom vorherigen Beispiel effizienter schreiben

    N = 100

    primzahlen = []

    for kandidat in range(2,N):        
        for teiler in range(2, kandidat-1):
            if kandidat % teiler == 0:
                hat_teiler = True
                break
        else:    
            primzahlen.append(kandidat)

    print(primzahlen)

## Schleifen 6 - enumerate, Tupel 
Tupel können über runde Klammern gebildet werden. Tupel können auch wieder in einzelne Variablen entpackt werden. Die **enumerate**-Funktion nimmt eine Liste (eigentlich einen Iterator, das ist ein technisches Detail) und erzeugt für jedes Element ein Tupel bestehend aus dem Index und dem Element selbst. 

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

## Schleifen 7 - Das Sieb des Eratosthenes
Das Sieb des Eratosthenes ist eine seid langem bekannte Methode um
Primzahlen zu bestimmen. Man beginnt mit einer Liste aller Zahl (Kandidaten) zwischen 2 und der größten Zahl an der man interessiert ist. Nun wiederholt man das Folgende solange wie noch Kandidaten übrig sind:

Die erste nicht gestrichene Zahl in der Liste ist eine Primzahl und wird aus der Liste gestrichen. Alle Vielfachen dieser Zahl sind keine Primzahlen, werden also ebenfalls gestrichen. Am Ende bleiben nur Primzahlen übrig. 


    # Das Sieb des Eratosthenes
    # Wir beginnen mit allen Zahlen zwischen 2 und 1000
    sieb = list(range(2,1000))

    # Am Anfang kennen wir keine Primzahlen
    primzahlen = []

    # Solange noch Kandidaten übrig sind
    while len(sieb) > 0:
        # Die erste Zahl in unserer Liste ist eine Primzahl
        # weil sie von keiner vorherigen Zahl geteilt wurde
        primzahl = sieb[0]
        primzahlen.append(primzahl)

        # "Siebe" nun alle Zahlen aus die von "primzahl" geteilt werden
        sieb = [x for x in sieb if x%primzahl!=0]    

    print(primzahlen)

# Lektion 5 - Funktionen
Alle folgenden Programme finden Sie in dem Ordner **lektion5**

## Funktionen
Wiederkehrender Programmcode mit in sich abgeschlossener Funktionalität kann in eigene Funktionen ausgegliedert werden. Dazu wird das **def**-Statement verwendet. Funktionen erhalten einen eigenen Namen damit diese später ausgeführt (aufgerufen) werden können. Der Programmcode der Funktion ist, ähnlich wie bei Schleifen und Bedinungen rechts eingerückt.

    def hallo():
        print("Hallo Welt")

    hallo()
    hallo()

## Funktionen 2 - Argumente und Rückgabewerte
Funktionen können Argumente entgegennehmen und innerhalb der Funktion verwenden um etwas zu berechnen. Mit Hife des **return**-Statements kann ein Wert zurückgegeben werden. 

    def addiere(x, y):
        return x + y

    a = 3
    b = 4
    print(f"Die Summe aus {a} und {b} ist {addiere(a,b)}")

## Funktionen 3 - Mehrere Rückgabewerte
Mit Hilfe von Tupeln können auch mehrere Werte zurückgegeben werden

    def berechne(x, y):
      return (x + y), (x - y), (x * y), (x / y)

    a = 3
    b = 4

    summe, differenz, produkt, quotient = berechne(a, b)
    print(f"Die Summe aus {a} und {b} ist {summe}")
    print(f"Die Differenz aus {a} und {b} ist {differenz}")
    print(f"Das Produkt aus {a} und {b} ist {produkt}")
    print(f"Der Quotient von {a} und {b} ist {quotient}")

## Funktionen 4 - Fibonnaci
Funktionen können auch Listen zurückgeben


    def fibonnaci(n):
        result = []
        a, b = 0,1
        
        while a < n:
            result.append(a)
            a, b = b, a+b
        
        return result

    print(fibonnaci(1000))


    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

## Funktionen 5 - Default-Argumente
Argumente können Standardwerte haben die verwendet werden wenn keine anderen angegeben werden. 

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

# Funktionen 6 - Funktionen als Argumente
Funktionen können selbst wieder Argumente einer Funktion sein.

    def highlight(x):
        print(f" --> {x} <--")

    def normal(x):
        print(x)

    def ausgabe(formater, n=10):
      for x in range(n):
        formater(x)

    ausgabe(normal)
    ausgabe(highlight, n=5)
    
# Funktionen 7 - Lambda-Funktionen, Map und Filter
Mit dem **lambda**-Statement lassen sind namenlose Funktionen definieren die überall dort verwendet können, wo eine Funktion benötigt wird. Das **map**-Keyword nimmt eine Funktion und eine Liste (Iterator) und wendet die Funktion auf jedes Element der Liste an. Die **filter**-Funktion wendet die Funktion ebenfalls auf jedes Element der Liste an und behält nur diejeniger Werte, bei denen die Funktion **True** liefert.

    # Wir starten mit allen Zahlen zwischen 1 und 10 (inklusive)
    zahlen = range(1,11)

    # Wir bilden jede Zahl auf ihr Quadrat ab
    quadratzahlen = map(lambda x: x**2, zahlen)

    # Filtere die ungeraden heraus
    gerade = filter(lambda x: x%2==0, quadratzahlen)

    # Erzeuge eine Liste und gib diese aus
    print(list(gerade))


    [4, 16, 36, 64, 100]

## Dictionaries    
Dictionaries speichern ungeordnete Key-Value Paare

    student = {
        "name": "Twenty 4 Tim",
        "matrikelnummer": 392918,
        "studiengang": "Deutsch, Englisch auf Lehramt"
    }

Werte können anhand ihres keys identifiziert werden

    print(student["name"])

Neue Werte können ebenfalls hinzugefügt werden

    student["geburtsdatum"] = "14.09.2000"

Mit dem *in* Keyword kann überprüft werden ob es einen bestimmten Key gibt

    print("name" in student)
    print("geschlecht" in student)

Die Funktion "items" liefert alle Key/Value Paare als Tupel

    for key, value in student.items():
        print(f"{key:20}: {value}")

# Lektion 6 - Module

## Import 
Bei größeren Projekten kann es Sinn machen den Code in mehrere, in sich geschlossene Module aufzuteilen. Die Datei *bibliothek.py* enthält eine Funktionsdefinition für die Fakultät aber keine direkten Anweisungen. Würde man diese Datei ausführen würde nichts passieren

    python lektion6/bibliothek.py

In der Datei *import.py* wird die Bibliothek mit dem **import**-Keywort importiert (eingebunden)    

    import bibliothek
    print(bibliothek.fakultaet(5))

Alle Definitionen aus *bibliothek.py* sind dann auch in *import.py* verfügbar, allerdings mit einem *bibliothek.*-Präfix. Das vermeidet Namensdopplungen wenn die selbe Funktion in mehreren Modulen definiert wird. 

## Import 2 - Direkter Import
Es ist möglich nur bestimmte Definitionen aus einem Modul zu importieren. Diese landen dann direkt im eigenen Namespace, können also ohne vorangestelltes Präfix angesprochen werden. 
    
    from bibliothek import fakultaet
    print(fakultaet(5))

## PI Schätzen
Die Bibliotheken **math** und **random** bieten nützliche Funktionen für mathematisch Operationen und zum Erzeugen von Zufallszahlen. Mit diesen läßt sich ein Verfahren zum Schätzen der Kreiszahl PI implementieren (vgl. [https://de.serlo.org/mathe/2107/kreiszahl-pi])

Wir importieren die beiden Bibliotheken

    import math
    import random

Wir beginnen mit 10000 Punkten

    anzahlPunkte = 10000

Nun brauchen wir noch zwei Hilfsfunktionen

    # Erzeuge einen zufälligen Punkt im Einheitsquadar
    def zufaelliger_punkt():
      return (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))

    # Bestimme den Abstand eines Punktes zum Ursprung
    def abstand(punkt):
      return math.sqrt(punkt[0]**2 + punkt[1]**2)

Der eigentliche Algorithmus läßt sich dann schreiben als

    # Bestimme zufällige Punkte im Einheitsquadrat
    quadrat = [zufaelliger_punkt() for _ in range(anzahlPunkte)]

    # Bestimme die Teilmenge der Punkte die auch im Einheitskreis liegen
    kreis = [x for x in quadrat if abstand(x) < 1]

    # Schätze Pi anhand dieses Verhältnisses
    pi = len(kreis) / anzahlPunkte * 4

    # Bestimme den relativen Fehler gegenüber dem korrekten Wert
    fehler = 100.0 * (pi / math.pi - 1)

    # Gib das ganze nett aus
    print(f"{anzahlPunkte:6} Punkte: π ≈ {pi:.6f} ({fehler:6.2f}% Fehler)")

## NumPy und PyPlot
Die **NumPy**-Bibliothek enthält viele nützliche Methoden für Vektormathematik. **PyPlot** bietet einfache Möglichkeiten Daten grafisch zu visualisieren

Zunächst importieren wir die beiden Bibliotheken
Beachte das wir mit dem **as**-Keyword neue Namen vergeben um
weniger Schreibarbeit im eigentlichen Code zu haben

    import numpy as np
    from matplotlib import pyplot as plt

Mit **np.linspace** können wir einen Vektor erzeugen der 9 gleichmäßig verteilte Werte zwischen -3.0 und +3.0 (inklusive) enthält. 

    x = np.linspace(-3.0, 3.0, 9)
    print(x)

    [-3.   -2.25 -1.5  -0.75  0.    0.75  1.5   2.25  3.  ]

Nun können wir die Werte komponentenweise quadrieren
    
    y = x**2
    print(y)

    [9.     5.0625 2.25   0.5625 0.     0.5625 2.25   5.0625 9.    ]


**PyPlot** erlaubt uns nun diese (X/Y)-Paare zu plotten

    plt.plot(x,y)
    plt.show()

# Pillow - Image Library
Die Pillow Bibliothek erlaubt es Bilder zu öffnen. Diese können
dann in ein NumPy Array umgewandelt und manipuliert werden. Wer braucht schon Photoshop wenn er Python hat? 

    from PIL import Image
    import numpy as np
    from matplotlib import pyplot as plt

    CONTRAST = 0.5
    BRIGHTNESS = -0.2

    img = Image.open("lektion6/forest.jpeg")
    img2 = np.array(img) / 255
    img2 = 0.5 + (img2 - 0.5) * (1.0 + CONTRAST)
    img2 = np.clip(img2 + BRIGHTNESS, 0.0, 1.0)
    img2 = Image.fromarray(np.uint8(img2 * 255))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.show()

# Interaktive GUI mit PyPlot
PyPlot erlaubt es einfache interaktive Widgets hinzuzufügen. Für diese können s.g. "callbacks" registriert werden, also Funktionen die immer dann aufgerufen werden wenn der Nutzer mit dem Widget interagiert. Damit läßt sich einfache Interaktivität in eine Python-Applikation einbauen. 

Für komplizierte GUIs gibt es dedizierte Libraries (z.B. PyQt5)

# Lektion 7 - Klassen
Klassen sind eine gute Möglichkeit Daten und auf diesen Daten operierende Methoden (Funktionen) zu bündeln.
Damit lassen sich auch komplexe Datenhierachien elegant abbilden. Klassen werden durch das **class**-Keyword deklariert und haben, ähnlich wie Funktionen, einen Namen.

## Klassen - Grundlagen
    class Professor:
        def __init__(self, name, fachgebiet):
            self.name = name
            self.fachgebiet = fachgebiet

Die spezielle Methode *\_\_init\_\_* ist der so genannte Konstruktor. Sie wird immer dann aufgerufen wenn eine neue Instanz dieser Klasse erzeugt werden soll, also z.B. so:

    prof = Professor("Dennis Müller", "Künstliche Intelligenz")

Der Parameter *self* referenziert dabei das Objekt selbst und erlaubt somit die Daten zu manipulieren. Neben dem Konstruktor können Klassen auch normale Methoden haben

    class Universitaet:
        def __init__(self, name):
            self.name = name
            self.professoren = []

        def berufen(self, prof):
            self.professoren.append(prof)

    hsd = Universitaet("Hochschule Duesseldorf")
    hsd.berufen(Professor("Dennis Müller", "Künstliche Intelligenz"))
    hsd.berufen(Professor("Florian Huber", "Data Science"))

# Klassen - Vererbung
Eine Klasse kann von einer anderen Klasse "erben". Die neue Klasse erhält damit
alle Variablen und Methoden der alten Klasse und kann diese verändern. Bei Methoden spricht man in diesem Zusammenhang auch von "überschreiben". 
Damit kann gemeinsames Verhalten in der s.g. Elternklasse gesammelt werden während spezialisiertes Verhalten in der s.g. Kindklasse implementiert wird. 

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

Mit dieser Klassenhierachie kann das folgende Programm ausgeführt werden
    
    dennisMueller = Professor("Dennis Müller", "Künstliche Intelligenz")

    pruefungen = []
    pruefungen.append(MuendlichePruefung("Elektrotechnik II", dennisMueller, 1.0, "Jochen Feitsch"))
    pruefungen.append(SchriftlichePruefung("Mathematik I", dennisMueller, 1.3, 120))

    for pruefung in pruefungen:
        pruefung.print()


    Mündliche Prüfung: Elektrotechnik II bei Dennis Müller, Note: 1.0. Beisitzer Jochen Feitsch    
    Schriftliche Prüfung: Mathematik I bei Dennis Müller, Note: 1.3. Länge 120 Minuten

## Ausnahmen - Exceptions
Exceptions ermöglichen es den normalen Kontrollfluß des Programms zu unterbrechen. 
Damit können ausserordentliche Fehler angezeigt werden die einer gesonderte Behandlung bedürfen. 

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

## Iteratoren
Iteratoren sind spezielle Klassen welche die s.g. *\_\_next\_\_* Methode implementieren. Iteratoren erlauben über Container oder andere (teilweise auch abstrakte) Mengen zu iterieren. Dabei wird die *\_\_next\_\_* Methode solange aufgerufen bis die *StopIteration*-Exception geworfen wird. Die Rückgabewerte der *\_\_next\_\_* Methode bilden die einzelnen Elemente des Iterators. Damit eine Klasse einen Iterator überhaupt erst erzeugen kann muß diese die *\_\_iter\_\_* Methode implementieren. Dabei ist es erlaubt hier einfach "self" zurückzugeben sofern die eigene Klasse *\_\_next\_\_* implementiert. 

    class Fibonnaci:
        def __init__(self, maximum):
            self.a, self.b = 1, 1
            self.maximum = maximum
        
        def __iter__(self):
            return self
        
        def __next__(self):
            self.a, self.b = self.a + self.b, self.a
            
            if self.a > self.maximum:
                raise StopIteration()
            
            return self.a

Mit dieser Klasse kann nun über die Fibonnaci Zahlen iteriert werden

    for i in Fibonnaci(20):
        print(i)


    2
    3
    5
    8
    13

Die **list** Funktion iteriert dabei vollständig über einen Iterator und fügt die Elemente alle in eine Liste ein

    lst = list(Fibonnaci(1000))    
    print(lst)


    [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]