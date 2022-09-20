import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy.random as rnd
import numpy as np
import random
import scipy as sp
import scipy.sparse as sparse
from time import time

# конструирование графа через заданную матрицу

N = 100 # макс чило вершин

def graph_constr(matrix):
    """
    конструирование графа со взвешенными ребрами по заданной матрице смежности
    :param matrix: матрица, где элемент x_ij -- вес ребра между вершинами i и j
    :return: граф
    """
    G = nx.Graph()
    size = matrix.shape[0]
    vertices = range(1,size+1)
    node_list = list(map(str,vertices[:size]))
    for node in node_list:
        G.add_node(node)
    for i in range(size):
        for j in range(size):
            if i < j and matrix[i][j] != 0:
                G.add_edge(node_list[i],node_list[j],weight=matrix[i][j])
    return G

def graph_constr_mrw(matrix, max_weight=11, seed=1234):
    """
    конструирование графа со взвешенными вершинами через матрицу смежности. Веса определяются рандомно через seed
    :param matrix: матрица смежности
    :param max_weight: максимальный вес вершины
    :param seed: определяет набор весов генерируемых вершин
    :return: граф
    """
    G = nx.Graph()
    size = matrix.shape[0]
    vertices = range(1, size + 1)
    node_list = list(map(str, vertices[:size]))
    rnd.seed(seed)
    for node in node_list:
        G.add_node(node, weight=np.random.randint(max_weight))
    for i in range(size):
        for j in range(size):
            if i < j and matrix[i][j] != 0:
                G.add_edge(node_list[i], node_list[j])
    return G

def graph_constr_mw(matrix, weights):
    """
    конструирование графа по заданной матрице смежности и набором весов для каждой вершины
    :param matrix: матрица смежности
    :param weights: список весов вершие
    :return: граф
    """
    G = nx.Graph()
    size = matrix.shape[0]
    vertices = range(1,size+1)
    node_list = list(map(str,vertices[:size]))
    for i in range(len(node_list)):
        G.add_node(node_list[i], weight=weights[str(int(i+1))])
    for i in range(size):
        for j in range(size):
            if i < j and matrix[i][j] != 0:
                G.add_edge(node_list[i],node_list[j])
    return G

# генерация рандомной матрицы для графа

def rand_graph_matrix(vert_num, edge_num, max_weight, seed=1234):
    """
    генерация рандомной матрицы смежности для графа со взвешенными ребрами
    :param vert_num: количество вершин
    :param edge_num: количество ребер
    :param max_weight: максимальный вес ребра
    :param seed: определяет набор весов для ребер
    :return: матрица смежности
    """
    a = nx.gnm_random_graph(vert_num, edge_num, seed=seed, directed=False)
    adj = nx.adjacency_matrix(a, nodelist=None, weight='weight')
    a_ = adj.A

    rnd.seed(seed)
    for i in range(vert_num):
        for j in range(vert_num):
            if j > i:
                if a_[i][j] != 0:
                    a_[i][j] = np.random.randint(max_weight)
    return np.array(a_ + a_.T)


def rand_graph_matrix_wv(vert_num, edge_num, seed=1234):
    """
    генерация матрица смежности для графа со взвешенными вершинами
    :param vert_num: количество вершин
    :param edge_num: количество ребер
    :param seed:
    :return: определяет набор весов генерируемых вершин
    """
    a = nx.gnm_random_graph(vert_num, edge_num, seed=seed, directed=False)
    adj = nx.adjacency_matrix(a, nodelist=None)
    a_ = adj.A
    return np.array(a_)


# получение матрицы графа

def get_graph_matrix(g):
    """
    получение матрицы смежности графа
    :param g: граф
    :return: матрица
    """
    adj = nx.adjacency_matrix(g, nodelist=None, weight='weight')
    size = g.number_of_nodes()
    mat = np.zeros((size, size))
    vertices = range(1, size + 1)

    info = nx.get_edge_attributes(G2, 'weight')

    for key, value in info.items():
        i, j = key
        i, j = int(i) - 1, int(j) - 1
        mat[i][j] = mat[j][i] = value
    return mat

# получение информации о графе

def get_graph_info_wv(g):
    """
    получение информации о графе
    :param g: граф
    :return: матрица смежности и веса вершин
    """
    adj = nx.adjacency_matrix(g, nodelist=None, weight='weight')
    mat = adj.A
    weights = nx.get_node_attributes(g, 'weight')

    return mat, weights


# отрисовка взвешенного графа

def plot_weighted_graph(graph):
    """
    отрисовка графа со взвешенными ребрами
    :param graph: граф
    """
    G = graph
    node_list = list(map(str, list(G.nodes)))
    #print(node_list)
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color='pink', node_size=500)

    # If you want, add labels to the nodes
    labels = {}
    for node_name in node_list:
        labels[str(node_name)] = str(node_name)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)

    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges))
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

    # Plot the graph
    plt.axis('off')
    # plt.title('title')
    # plt.savefig("pic_name.png")
    plt.show()


def plot_weighted_graph_wv(graph):
    """
    отрисовка графа со взвешенными вершинами
    :param graph: граф
    """
    G = graph
    node_list = list(map(str, list(G.nodes)))
    pos = nx.circular_layout(G)
    weights = nx.get_node_attributes(G, 'weight')
    nx.draw_networkx_nodes(G, pos, node_color='pink', node_size=500)

    # If you want, add labels to the nodes
    labels = {}
    for i in range(len(node_list)):
        labels[str(node_list[i])] = str(node_list[i]) + " (" + str(weights[str(node_list[i])]) + ")"
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges))
    plt.axis('off')
    plt.show()

# визуализация решения

def solution_visualization(info):
    """
    визуализация решения интервальной раскраски для графа со взвешенными ребрами
    :param info: массив из массивов вида [вершина1, вершина2, время старта, время конца],
    где вершина1 и вершина 2 определяют ребро
    """
    y_ = len(info) + 1
    plt.plot(0, y_, marker='o', color='white')  # обманка, чтобы верхняя грань графика не налезала на надписи
    for edge in info:
        x = [edge[-1], edge[-1] + edge[-2]]  # начало отрезка в стартовой точке, конец = стартовая точка + вес ребра
        y_ -= 1  # самые длинные ребра будут сверху. шаг между ребрами = 1
        y = [y_, y_]
        plt.plot(x, y, marker='o', color='pink')
        c = 0.2
        plt.text(x[0], y[0] + c, edge[0], fontsize=14)  # подписываем левый конец отрезка
        plt.text(x[1], y[1] + c, edge[1], fontsize=14)  # подписываем правый конец отрезка

    plt.show()

def solution_visualization_wv(info):
    """
    визуализация решения интервальной раскраски для графа со взвещенными вершинами
    :param info: массив из массивов вида [номер вершины, старт, окончание]
    """
    plt.figure(figsize=(10,8))
    y_ = info.shape[0]+1
    plt.plot(0, y_, marker = 'o', color='white') # обманка, чтобы верхняя грань графика не налезала на надписи
    for node in info:
        x = [node[1],node[2]] # начало отрезка в стартовой точке, конец = стартовая точка + вес ребра
        y_ -= 1 # самые длинные ребра будут сверху. шаг между ребрами = 1
        y = [y_, y_]
        plt.plot(x, y, marker = 'o', color='pink')
        plt.text((x[0]+x[1])/2, y[0]+0.1, "v #"+ str(int(node[0])), fontsize=10) # подписываем номер вершины
    #plt.savefig("sol.png")
    plt.show()
