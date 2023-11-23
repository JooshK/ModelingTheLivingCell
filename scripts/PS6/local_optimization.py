import numpy as np
import pandas as pd
from PS5.langevian_protein import calculate_configuration_force


def process_line(line):
    parts = line.split()
    label = parts[0]
    coordinates = [float(val) for val in parts[1:]]
    return label, coordinates


def process_file(path):
    data = []
    with open(path, 'r') as file:
        next(file)
        for line in file:
            label, coordinates = process_line(line)
            data.append([label] + coordinates)

    return pd.DataFrame(data, columns=['H/P', 'x', 'y', 'z'])


def steepest_descent(data: pd.DataFrame, step_rate):
    coordinates = data[['x', 'y', 'z']].to_numpy()
    residues = data['H/P'].to_numpy()

    while True:
        energy, gradient = calculate_configuration_force(coordinates, residues, k=20, l=1)
        new_coordinates = coordinates + (gradient*step_rate)

        if np.all(new_coordinates == coordinates):
            break

        coordinates = new_coordinates
        print(energy)

    return energy, coordinates


