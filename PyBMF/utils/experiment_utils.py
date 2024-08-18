import pickle, os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def get_model_by_path(path_list):
    model_list = []

    for i, path in enumerate(path_list):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        model_list.append(model)

    return model_list



def get_model_by_time(root='../saved_models/', model_name='Asso', time_start="24-06-03 04:43", time_end="24-06-03 07:00"):

    extension = model_name + '.pickle'

    path_list = []

    time_start = datetime.strptime(time_start, '%y-%m-%d %H:%M')
    time_end = datetime.strptime(time_end, '%y-%m-%d %H:%M')

    for file in os.listdir(root):
        file_path = os.path.join(root, file)
        t = datetime.fromtimestamp(os.path.getctime(file_path))

        if time_end >= t >= time_start and file.endswith(extension):
            path_list.append(file_path)

    print(path_list)

    model_list = get_model_by_path(path_list)

    return model_list

