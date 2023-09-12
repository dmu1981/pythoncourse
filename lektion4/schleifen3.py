FSK = 18

namen = ["Anton", "Beate", "Clara"]
alter = [17, 19, 22]

for tuple in zip(namen, alter):
    if tuple[1] < FSK:
        continue
    
    print(f"{tuple[0]} ist {tuple[1]} Jahre alt")