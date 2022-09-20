import numpy as np
import pandas as pd
#import os

from funcs import graph_constr_mw, graph_constr_mw_


def bench_graph(N):
    #path to test N

    # вставить свой путь
    path=f"Contest Voronovo Archiv//inputs//{N}//"

    links = pd.read_csv(path+"links.csv")
    newrouting = pd.read_csv(path+"newrouting.csv")
    nodesinfo = pd.read_csv(path+"nodesinfo.csv")

    newrouting = newrouting.sort_values("path_id") # — отсортированные пути
    newrouting = newrouting.to_numpy()

    routes = np.unique(newrouting.T[0]) # список ID путей — возвращает отсортированный массив

    adj_matrix = np.zeros((routes.size, routes.size))

    # массив ребер (линков) для каждого пути

    routes_links = []  # массив с ребрами для каждого пути
    routes_weights = []  # массив с весом путей

    N = newrouting[0][0]
    # routes = [] # список из ID путей, чтобы сопоставлять
    # routes.append(N)

    routes_weights.append(newrouting[0][-1])
    temp = []

    for line in newrouting:
        if line[0] == N:
            temp.append(line[1])
        else:
            routes_links.append(temp)
            temp = [line[1]]
            N = line[0]
            # routes.append(N)
            routes_weights.append(line[-1])

    # добавляем инфу по последнему пути
    routes_links.append(temp)
    temp = [line[1]]
    # N = line[0]
    # routes.append(N)
    routes_weights.append(line[-1])

    for i in range(len(routes)):
        for j in range(len(routes)):
            # print(i, j)
            if adj_matrix[i][j] == 0:
                if np.any(np.in1d(routes_links[i], routes_links[j])):  # если у 2 путей совпадают хотя бы 1 ребро
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1



    weights = {}
    for i in range(len(routes)):
        weights[str(i)] = routes_weights[i] # делаем вершины 0 ... N

    #return graph_constr_mw(adj_matrix, weights)
    return graph_constr_mw_(adj_matrix, weights)

# ВОЗВРАЩАЕМСЯ К ИСХОДНЫМ id ДЛЯ ПУТЕЙ
def back_to_routes_ID(res, routes):
    for i in range(len(res)):
        res[i][0] = routes[i]
    return res