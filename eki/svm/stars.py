import csv
import numpy as np
from enum import IntEnum

def load_stars(split=False):
    data = None
    csv_reader = csv.reader(open("pulsar_stars.csv", "rt"))
    next(csv_reader)
    for line in csv_reader:
        x = np.array([[
            float(line[0]),
            float(line[1]),
            float(line[2]),
            float(line[3]),
            float(line[4]),
            float(line[5]),
            float(line[6]),
            float(line[7]),
            float(line[8]),
            ]])
        if data is None:
            data = x
        else:
            data = np.concatenate((data, x))

    if split:
        #np.random.shuffle(data)
        train_size = data.shape[0] // 2
        return data[:train_size], data[train_size:]
    else:
        return data

