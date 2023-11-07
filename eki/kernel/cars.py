import csv
import numpy as np
from enum import IntEnum

class Features(IntEnum):
    Bias = 0,
    Year = 1,           # Baujahr des Fahrzeugs (z.b. 2013)
    KM = 2,             # Kilometer Driven (z.B. 12000)
    Fuel = 3,           # Antriebstyp (0 = Benzin, 1 = Diesel)
    Seller = 4,         # Verkäufer (0 == Händler, 1 = Privat)
    Transmission = 5,   # Getriebe (0 == Schaltung, 1 = Automatik)
    Price = 6           # Verkaufspreis in tausend Euro (z.B. 8.3)

def load_cars_data(split=False):
    data = None
    csv_reader = csv.reader(open("cars.csv", "rt"))
    next(csv_reader)
    for line in csv_reader:
        year, price, kmdriven = float(line[1]), float(line[2]), float(line[4])
        fuel = 0 if line[5] == 'Petrol' else 1
        seller = 0 if line[6] == 'Dealer' else 1
        transmission = 0 if line[7] == 'Manual' else 1

        x = np.array([[1, year, kmdriven, fuel, seller, transmission, price]])
        if data is None:
            data = x
        else:
            data = np.concatenate((data, x))

    if split:
        np.random.shuffle(data)
        train_size = data.shape[0] * 4 // 5
        return data[:train_size], data[train_size:]
    else:
        return data

