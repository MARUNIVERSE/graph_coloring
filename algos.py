import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy.random as rnd
import numpy as np
import random
import scipy as sp
import scipy.sparse as sparse
from time import time


# LF-алгоритм для графов со взевешенными ребрами использованием сортировки numpy (код Дани)

def LF_new(g):
    edges_dict = nx.get_edge_attributes(g, 'weight')  # получаем инфу в виде словаря

    # делаем из словаря список
    edges = []
    for key, value in edges_dict.items():
        edges.append((int(key[0]), int(key[1]), value))
    edges = np.array(edges, dtype=[('edge1', int), ('edge2', int), ('weight', int)])

    start = time()

    sorted_graph = np.sort(edges, order='weight')[::-1]
    # print(sorted_graph)
    # earliest_time = 0
    N = len(edges)
    res = np.zeros((N, 4))
    for i in range(N):
        res[i][0] = sorted_graph[i][0]
        res[i][1] = sorted_graph[i][1]
        earliest = 0
        for p in range(i + 1):
            if res[p][0] == sorted_graph[i][0]:
                earliest = max([earliest, res[p][3]])
            elif res[p][1] == sorted_graph[i][0]:
                earliest = max([earliest, res[p][3]])
            elif res[p][0] == sorted_graph[i][1]:
                earliest = max([earliest, res[p][3]])
            elif res[p][1] == sorted_graph[i][1]:
                earliest = max([earliest, res[p][3]])
        res[i][2] = earliest
        res[i][3] = earliest + sorted_graph[i][2]
    end = time()
    return res, end - start

# LF-алгоритм для графов со взвешенными вершинами

def LF_wv(graph):
    nodes_dict = nx.get_node_attributes(graph, 'weight')  # получаем инфу в виде словаря
    adj = np.array(nx.adjacency_matrix(graph).A)
    if adj.shape[0] != len(nodes_dict.keys()):
        print("smth bad happened, idk")
    # делаем из словаря список
    nodes = []
    for key, value in nodes_dict.items():
        nodes.append((int(key), value))
    nodes = np.array(nodes, dtype=[('node', int), ('weight', int)])

    start = time()

    sorted_graph = np.sort(nodes, order='weight')[::-1]
    # print(sorted_graph)
    # earliest_time = 0
    N = len(nodes)
    res = np.zeros((N, 3))
    for i in range(N):
        res[i][0] = sorted_graph[i][0]
        earliest = 0
        for p in range(i):
            if adj[int(res[i][0]) - 1, int(res[p][0]) - 1] > 0:
                earliest = max(earliest, res[p, 2])
        res[i][1] = earliest
        res[i][2] = earliest + sorted_graph[i][1]
    end = time()
    opt = max(res[::, 2])
    return res, end - start, opt


def reg_coloring_intervals(graph):
    nodes_dict = nx.get_node_attributes(graph, 'weight')

    adj = np.array(nx.adjacency_matrix(graph).A)
    nodes = []
    for key, value in nodes_dict.items():
        nodes.append((int(key), value, np.sum(adj[::, int(key) - 1]), 0))
    nodes = np.array(nodes, dtype=[('node', int), ('weight', int), ('degree', int), ('color', int)])
    opt = 0
    N = len(nodes)
    res = np.zeros((N, 3))
    color_times = {0: 0}
    start = time()
    sorted_nodes = np.sort(nodes, order='degree')[::-1]
    # sorted_nodes = nodes
    for i in range(N):
        banned = set()
        for j in range(N):
            if adj[sorted_nodes[i][0] - 1, sorted_nodes[j][0] - 1] > 0:
                banned.add(sorted_nodes[j][3])
        color = 1
        while color in banned:
            color += 1
        sorted_nodes[i][3] = color
        # print(sorted_nodes[i][0],color,banned)
        tmp = color_times.get(sorted_nodes[i][3])
        if tmp == None:
            color_times.update({sorted_nodes[i][3]: sorted_nodes[i][1]})
        else:
            color_times.update({sorted_nodes[i][3]: max(tmp, sorted_nodes[i][1])})
    # print(sorted_nodes)
    opt = sum(color_times.values())
    color_start = np.zeros((len(color_times), 2))
    for i in range(N):
        c = sorted_nodes[i][3]
        res[i, 0] = sorted_nodes[i][0]
        res[i, 1] = color_start[c, 0]
        res[i, 2] = res[i, 1] + sorted_nodes[i][1]
        if color_start[c, 1] < 1:
            o = color_times.get(c)
            color_start[c, 1] += 1
            color_start[::, 0] += o
            color_start[::, 0] -= (color_start[::, 1] > 0) * o
            # print(color_start.T, res[i])
    end = time()
    # print(color_start.T)
    # print(color_times)
    return res, end - start, opt


def reg_coloring_intervals(graph):
    nodes_dict = nx.get_node_attributes(graph, 'weight')

    adj = np.array(nx.adjacency_matrix(graph).A)
    nodes = []
    for key, value in nodes_dict.items():
        nodes.append((int(key), value, np.sum(adj[::, int(key) - 1]), 0))
    nodes = np.array(nodes, dtype=[('node', int), ('weight', int), ('degree', int), ('color', int)])
    opt = 0
    N = len(nodes)
    res = np.zeros((N, 3))
    color_times = {0: 0}
    start = time()
    sorted_nodes = np.sort(nodes, order='degree')[::-1]
    # sorted_nodes = nodes
    for i in range(N):
        banned = set()
        for j in range(N):
            if adj[sorted_nodes[i][0] - 1, sorted_nodes[j][0] - 1] > 0:
                banned.add(sorted_nodes[j][3])
        color = 1
        while color in banned:
            color += 1
        sorted_nodes[i][3] = color
        # print(sorted_nodes[i][0],color,banned)
        tmp = color_times.get(sorted_nodes[i][3])
        if tmp == None:
            color_times.update({sorted_nodes[i][3]: sorted_nodes[i][1]})
        else:
            color_times.update({sorted_nodes[i][3]: max(tmp, sorted_nodes[i][1])})
    # print(sorted_nodes)
    opt = sum(color_times.values())
    color_start = np.zeros((len(color_times), 2))
    for i in range(N):
        c = sorted_nodes[i][3]
        res[i, 0] = sorted_nodes[i][0]
        res[i, 1] = color_start[c, 0]
        res[i, 2] = res[i, 1] + sorted_nodes[i][1]
        if color_start[c, 1] < 1:
            o = color_times.get(c)
            color_start[c, 1] += 1
            color_start[::, 0] += o
            color_start[::, 0] -= (color_start[::, 1] > 0) * o
            # print(color_start.T, res[i])
    end = time()
    # print(color_start.T)
    # print(color_times)
    return res, end - start, opt