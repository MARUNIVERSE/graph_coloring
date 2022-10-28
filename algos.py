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
        earliest = 0 # время старта для i вершины
        for p in range(i):
            if adj[int(res[i][0]) - 1, int(res[p][0]) - 1] > 0: # если в рассмотренном множестве мы нашли смежную с нашей вершину
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


def tree_coloring(graph):
    nodes_dict = nx.get_node_attributes(graph, 'weight')

    adj = np.array(nx.adjacency_matrix(graph).A)
    nodes = []
    for key, value in nodes_dict.items():
        nodes.append((int(key), value))
    nodes = np.array(nodes, dtype=[('node', int), ('weight', int)])
    sorted_nodes = np.sort(nodes, order='weight')[::-1]
    N = nodes.size
    colored_set = set()
    colored = []
    col = 0
    i = 0
    while len(colored) < N:
        while sorted_nodes[i][0] in colored_set:
            i += 1
        v0 = sorted_nodes[i][0]
        colored.append((v0, col))
        colored_set.add(v0)
        for k in range(N):
            v = sorted_nodes[k][0]
            A = v in colored_set
            B = adj[v - 1][v0 - 1] > 0
            if B or A:
                continue
            flag = True
            for p in colored:
                if p[1] == col:
                    if adj[p[0] - 1][v - 1] > 0:
                        flag = False
                        break
            if flag:
                colored.append((v, col))
                colored_set.add(v)
        col += 1
    res = np.zeros((N, 3))
    earliest = 0
    current = 0
    prev = 0
    max_t = 0
    # print(colored)
    # print(colored_set)
    for g in range(N):
        if prev == colored[g][1]:
            res[g][0] = colored[g][0]
            res[g][1] = current
            res[g][2] = nodes[colored[g][0] - 1][1] + current
            earliest = max(earliest, res[g][2])
        else:
            prev = colored[g][1]
            current = earliest
            res[g][0] = colored[g][0]
            res[g][1] = current
            res[g][2] = nodes[colored[g][0] - 1][1] + current
            earliest = max(earliest, res[g][2])
        #if earliest >= 319:
        #    return res[:g, ::], max_t
        max_t = earliest
    return res, max_t

def tree_coloring_mod_(g):
    coloring, _ = tree_coloring(g)
    coloring = sorted(coloring, key=lambda x: x[-1]-x[1], reverse=True) # сортируем вершины в порядке невозрастания
    for i in range(len(coloring)):
        coloring[i] = list(coloring[i]) + list(coloring[i][1:]) # добавляем новое время, определяющее старт и конец
    colors = sorted(list(set([x[1] for x in coloring]))) # определяем цвета в раскраске и их количество
    res = np.copy(coloring)

    vert_index = [x[0] for x in coloring]

    for i in range(len(colors)):
        if colors[i] == 0:
            continue # не можем сдвигать влево врешины, стартующие первыми

        prev_color_vertices = [x for x in coloring if x[1] == colors[i-1]] # рассматриваем вершины предыдщуего цвета
        cur_color_vertices = [x for x in coloring if x[1] == colors[i]] # рассматриваем вершины текущего цвета

        for j in range(len(cur_color_vertices)): # пытаемся сдвинуть текущие вершины
            n_j = g[str(int(cur_color_vertices[j][0]))]

            for k in range(len(prev_color_vertices)):
                t = vert_index.index(cur_color_vertices[j][0])
                if str(int(prev_color_vertices[k][0])) in n_j:
                    diff = coloring[t][1] - prev_color_vertices[k][-1] # на сколько нужно сдвинуть вершину
                    coloring[t][-1] -= diff
                    coloring[t][-2] -= diff
                    #print(f'diff={diff}')
                # сдвинули вершину -- переходим дальше (нет смысла рассматривать возможность большего сдвига, иначе будет наложение с первой смежной вершиной)
                    break

    result = [[x[0], x[3], x[4]] for x in coloring]
    return np.asarray(result), max([x[-1] for x in coloring])


def tree_coloring_mod(g):
    coloring, _ = tree_coloring(g)
    coloring = sorted(coloring, key=lambda x: x[-1] - x[1], reverse=True)  # сортируем вершины в порядке невозрастания
    for i in range(len(coloring)):
        coloring[i] = list(coloring[i]) + list(coloring[i][1:])  # добавляем новое время, определяющее старт и конец
    colors = sorted(list(set([x[1] for x in coloring])))  # определяем цвета в раскраске и их количество
    res = np.copy(coloring)

    vert_index = [x[0] for x in coloring]

    for i in range(len(colors)):
        if colors[i] == 0:
            continue  # не можем сдвигать влево врешины, стартующие первыми

        prev_color_vertices = [x for x in coloring if x[1] == colors[i - 1]]  # рассматриваем вершины предыдщуего цвета
        cur_color_vertices = [x for x in coloring if x[1] == colors[i]]  # рассматриваем вершины текущего цвета

        for j in range(len(cur_color_vertices)):  # пытаемся сдвинуть текущие вершины
            n_j = g[str(int(cur_color_vertices[j][0]))]  # смежные вершины
            t = vert_index.index(cur_color_vertices[j][0])
            buf = []

            for k in range(len(prev_color_vertices)):
                if str(int(prev_color_vertices[k][0])) in n_j:
                    buf.append(prev_color_vertices[k])  # список смежных вершин с текущей

            max_rb = sorted(buf, key=lambda x: x[-1], reverse=True)[0][
                -1]  # определяем границу, куда можно максимально сдвинуть вершину
            diff = coloring[t][1] - max_rb  # на сколько нужно сдвинуть вершину
            coloring[t][-1] -= diff
            coloring[t][-2] -= diff
            # print(f'diff={diff}')
            # сдвинули вершину -- переходим дальше (нет смысла рассматривать возможность большего сдвига, иначе будет наложение с первой смежной вершиной)

    result = [[x[0], x[3], x[4]] for x in coloring]
    return np.asarray(result), max([x[-1] for x in coloring])

# Последняя версия сдвигов для жадных раскрасок (точно)
def sdvig_interv(g, strategy='largest_first'):
    coloring, _ = greedy(g, strategy=strategy)
    
    # print(coloring)
    # coloring = [[vertex_num, start, end]]
    coloring = sorted(coloring, key=lambda x: x[0]) # сортируем в порядке вершин: 1, 2, 3...
    # чтобы индекс в массиве совпадал с именем вершины
    for i in range(len(coloring)):
        coloring[i] = list(coloring[i]) + list(coloring[i][1:])  # добавляем новое время, определяющее старт и конец,
        # инициализируем старыми началом и концом
    coloring = np.array(coloring)

    # все вершины первого цвета отправляем в просмотренные -- с ними ничего нельзя сделать

    colors = sorted(list(set([x[1] for x in coloring])))  # определяем цвета в раскраске и их количество
    # подразумевается что интервалы построены по раскраске и вершины одного цвета имеют общую левую границу
    visited = set()# вершины левее текущего цвета
    for j in range(len(colors)-1): # индекс указывает на цвет, к которому сдвигаем
        cur_color_v = [int(x[0]) for x in coloring if x[1] == colors[j+1]] # смотрим вершины текущего цвета
        #print(cur_color_v)
        prev_color_v = [int(x[0]) for x in coloring if x[1] == colors[j]] # вершины предыдущего цвета
        visited.update(set(prev_color_v))# добавляем предыдущий цвет в пройденные вершины
        for i in cur_color_v:
            #print(visited, set(g[str(int(i))]))
            s = visited & set(map(int, g[str(int(i))])) # вершины смежные с текущей и расположенные левее
            #print(s,'adj to', i)
            max_v=0
            maxvid=0
            for v in s: # определяем наибольшую правую границу среди смежных вершин, к ней сдвигаемся
                if coloring[v,4]>max_v:
                    max_v = coloring[v,4]
                    maxvid = v
            #print(maxvid, max_v,'for', i)
            w = coloring[i,2]-coloring[i,1] # вес
            #print(i,'has weight',w)
            coloring[i,3] = max_v # новые начало и конец
            coloring[i,4] = max_v + w

    result = [[x[0], x[3], x[4]] for x in coloring]
    return np.asarray(result), max([x[-1] for x in coloring])

# Последняя версия для LF (точно)
def LF_sdvig(g):
    coloring, _, _ = LF_wv(g)
    # print(coloring)
    # coloring = [[vertex_num, start, end]]
    coloring = sorted(coloring, key=lambda x: x[0]) # сортируем в порядке вершин: 1, 2, 3...
    # чтобы индекс в массиве совпадал с именем вершины
    for i in range(len(coloring)):
        coloring[i] = list(coloring[i]) + list(coloring[i][1:])  # добавляем новое время, определяющее старт и конец,
        # инициализируем старыми началом и концом
    coloring = np.array(coloring)

    # все вершины первого цвета отправляем в просмотренные -- с ними ничего нельзя сделать

    colors = sorted(list(set([x[1] for x in coloring])))  # определяем цвета в раскраске и их количество
    # подразумевается что интервалы построены по раскраске и вершины одного цвета имеют общую левую границу
    visited = set()# вершины левее текущего цвета
    for j in range(len(colors)-1): # индекс указывает на цвет, к которому сдвигаем
        cur_color_v = [int(x[0]) for x in coloring if x[1] == colors[j+1]] # смотрим вершины текущего цвета
        #print(cur_color_v)
        prev_color_v = [int(x[0]) for x in coloring if x[1] == colors[j]] # вершины предыдущего цвета
        visited.update(set(prev_color_v))# добавляем предыдущий цвет в пройденные вершины
        for i in cur_color_v:
            #print(visited, set(g[str(int(i))]))
            s = visited & set(map(int, g[str(int(i))])) # вершины смежные с текущей и расположенные левее
            #print(s,'adj to', i)
            max_v=0
            maxvid=0
            for v in s: # определяем наибольшую правую границу среди смежных вершин, к ней сдвигаемся
                if coloring[v,4]>max_v:
                    max_v = coloring[v,4]
                    maxvid = v
            #print(maxvid, max_v,'for', i)
            w = coloring[i,2]-coloring[i,1] # вес
            #print(i,'has weight',w)
            coloring[i,3] = max_v # новые начало и конец
            coloring[i,4] = max_v + w

    result = [[x[0], x[3], x[4]] for x in coloring]
    return np.asarray(result), max([x[-1] for x in coloring])


# универсальная функция для обработки результатов раскраски
def sdvig_any(g, c): # на входе граф из N вершин и готовая раскраска np.array((N,3))
    coloring = c.copy()
    # print(coloring)
    # coloring = [[vertex_num, start, end]]
    coloring = sorted(coloring, key=lambda x: x[0]) # сортируем в порядке вершин: 1, 2, 3...
    # чтобы индекс в массиве совпадал с именем вершины
    for i in range(len(coloring)):
        coloring[i] = list(coloring[i]) + list(coloring[i][1:])  # добавляем новое время, определяющее старт и конец,
        # инициализируем старыми началом и концом
    coloring = np.array(coloring)

    # все вершины первого цвета отправляем в просмотренные -- с ними ничего нельзя сделать

    colors = sorted(list(set([x[1] for x in coloring])))  # определяем цвета в раскраске и их количество
    # подразумевается что интервалы построены по раскраске и вершины одного цвета имеют общую левую границу
    visited = set()# вершины левее текущего цвета
    for j in range(len(colors)-1): # индекс указывает на цвет, к которому сдвигаем
        cur_color_v = [int(x[0]) for x in coloring if x[1] == colors[j+1]] # смотрим вершины текущего цвета
        #print(cur_color_v)
        prev_color_v = [int(x[0]) for x in coloring if x[1] == colors[j]] # вершины предыдущего цвета
        visited.update(set(prev_color_v))# добавляем предыдущий цвет в пройденные вершины
        for i in cur_color_v:
            #print(visited, set(g[str(int(i))]))
            s = visited & set(map(int, g[str(int(i))])) # вершины смежные с текущей и расположенные левее
            #print(s,'adj to', i)
            max_v=0
            maxvid=0
            for v in s: # определяем наибольшую правую границу среди смежных вершин, к ней сдвигаемся
                if coloring[v,4]>max_v:
                    max_v = coloring[v,4]
                    maxvid = v
            #print(maxvid, max_v,'for', i)
            w = coloring[i,2]-coloring[i,1] # вес
            #print(i,'has weight',w)
            coloring[i,3] = max_v # новые начало и конец
            coloring[i,4] = max_v + w

    result = [[x[0], x[3], x[4]] for x in coloring]
    return np.asarray(result), max([x[-1] for x in coloring])

def Clique_sampling(graph, samples, seed=69):
    nodes_dict = nx.get_node_attributes(graph, 'weight')
    
    adj = np.array(nx.adjacency_matrix(graph).A)
    nodes = []
    for key, value in nodes_dict.items():
        nodes.append((int(key), value, np.sum(adj[::,int(key)]), -1))
    nodes = np.array(nodes, dtype=[('node', int),('weight', int), ('degree', int),('color',int)])
    N = len(nodes)
    rng = default_rng(seed)
    Cliset = []
    Startset = set()
    idx = 0
    while len(Startset) < N//samples: # выбираем вершины для построения подграфов
        if rng.random()>0.5:
            Startset.add(idx)
        idx+=1
        idx=idx%N
    print(N,Startset)
    while len(Startset)>0: # строим подграфы максимального веса
        nod = Startset.pop()
        C = []
        C.append(nodes[nod])
        Cand = set(map(int, graph[str(int(nod))])) # соседние вершины
        while len(Cand)>0:
            max_ = 0
            new = -1
            for i in Cand:
                inters = set(map(int, graph[str(int(i))])).intersection(Cand)
                inters.add(i)
                tmp = 0
                for j in inters:
                    tmp+=nodes[j]['weight']
                if tmp > max_:
                    new=i
                    max_=tmp
            if new==-1:
                break
            C.append(nodes[new])
            Cand.remove(new)
            Cand.intersection_update(set(map(int, graph[str(int(new))])))
        Cliset.append(C)
    l = 0
    for i in Cliset:
        if len(i)>l:
            l=len(i)
    for i in Cliset: # уравниваем размер подграфов
        while len(i) < l:
            i.append((-1,0,0,-1))
    Cliset = np.array(Cliset,dtype=[('node', int),('weight', int), ('degree', int),('color',int)])
    maxcli = 0
    clinum = 0
    for i in range(len(Cliset)): # ищем подграф максимального веса
        Cliset[i] = np.sort(Cliset[i], order='weight')[::-1]
        clisum = np.sum(Cliset[i]['weight'])
        if clisum>maxcli:
            maxcli = clisum
            clinum = i
    Sm = Cliset.T
    cost = 0
    for i in range(l):
        cost+=np.max(Sm[i,:]['weight'])
    print('lower bound estimation: ', cost) # оценка снизу
    Notincliset = set(range(N))
    Removed = set()
    major = 0
    for i in Cliset[clinum]: # составляем список вершин, не входящих в подграф
        if i['weight']>0:
            major+=i['weight']
            Notincliset.remove(i['node'])
    print('Heaviest Clique: ',major) # вес тяжелейшего подграфа
    for i in Notincliset:
        if nodes[i]['degree']<l:
            Removed.add(i)
    Notincliset.difference_update(Removed) # некоторые вершины не влияют на правую границу
    initColoring = []
    curcolor = 0
    for h in Cliset[clinum]: # красим полный подграф
        if h['weight']>0:
            h['color']=curcolor
            curcolor+=1
            initColoring.append([h])    
    Notincli = []
    for i in Notincliset:
        Notincli.append(nodes[i])
    Notincli = np.array(Notincli,dtype=[('node', int),('weight', int), ('degree', int),('color',int)])
    Notincli = np.sort(Notincli,order=['degree','weight'])[::-1]
    
    for nd in Notincli:
        done=False
        adj_v = set(map(int, graph[str(int(nd['node']))]))
        for trycolor in range(len(initColoring)):
            can = True
            for k in initColoring[trycolor]:
                if k['node'] in adj_v:
                    can=False
                    break
            if can:
                nd['color'] = trycolor
                initColoring[trycolor].append(nd)
                done=True
                break
        if done==False:
            nd['color'] = curcolor
            curcolor+=1
            initColoring.append([nd])
    Removed2 = Removed.copy()
    print(Removed)
    for nd in Removed:
        q = nodes[nd]
        adj_v = set(map(int, graph[str(int(nd))]))
        done = False
        for trycolor in range(len(initColoring)):
            can = True
            for vert in initColoring[trycolor]:
                if vert['node'] in adj_v:
                    can = False
            if can == True:
                q['color']=trycolor
                initColoring[trycolor].append(q)
                done = True
                break
        if done ==True:
            Removed2.remove(nd)
        else:
            print('removed not added :(')
    answer = np.zeros((N,3))
    left = 0
    iteration = 0
    print(initColoring)
    for i in range(len(initColoring)):
        for j in initColoring[i]:
            answer[iteration,::]=np.array([j['node'],left,left+j['weight']])
            iteration+=1
        left+=np.max(np.array(initColoring[i],
                              dtype=[('node', int),('weight', int), ('degree', int),('color',int)])['weight'])
    return answer, left


# исправленная версия
def greedy_mod(g, strategy='largest_first'):
    coloring, _ = greedy(g, strategy=strategy)

    coloring = sorted(coloring, key=lambda x: x[-1] - x[1], reverse=True)  # сортируем вершины в порядке неубывания
    for i in range(len(coloring)):
        coloring[i] = list(coloring[i]) + list(coloring[i][1:])  # добавляем новое время, определяющее старт и конец
    colors = sorted(list(set([x[1] for x in coloring])))  # определяем цвета в раскраске и их количество
    res = np.copy(coloring)

    vert_index = [x[0] for x in coloring]

    for i in range(len(colors)):
        if colors[i] == 0:
            continue  # не можем сдвигать влево врешины, стартующие первыми

        prev_color_vertices = [x for x in coloring if x[1] == colors[i - 1]]  # рассматриваем вершины предыдщуего цвета
        cur_color_vertices = [x for x in coloring if x[1] == colors[i]]  # рассматриваем вершины текущего цвета

        for j in range(len(cur_color_vertices)):  # пытаемся сдвинуть текущие вершины
            n_j = g[str(int(cur_color_vertices[j][0]))]  # смежные вершины
            t = vert_index.index(cur_color_vertices[j][0])
            buf = []

            for k in range(len(prev_color_vertices)):
                if str(int(prev_color_vertices[k][0])) in n_j:
                    buf.append(prev_color_vertices[k])  # список смежных вершин с текущей

            max_rb = sorted(buf, key=lambda x: x[-1], reverse=True)[0][-1]  # определяем границу, куда можно максимально сдвинуть вершину
            diff = coloring[t][1] - max_rb  # на сколько нужно сдвинуть вершину
            coloring[t][-1] -= diff
            coloring[t][-2] -= diff
            # print(f'diff={diff}')

    result = [[x[0], x[3], x[4]] for x in coloring]
    return np.asarray(result), max([x[-1] for x in coloring])

# алгоритм на основе жадной раскраски графа

from collections import defaultdict

def greedy(g, strategy='largest_first'):
    colors = nx.greedy_color(g, strategy=strategy) # словарь вершина : цвет
    number_of_colors = max(colors.values())+1
    vert = [[]]*number_of_colors # массив массивов вершин разных цветов вида вершина : вес вершины
    sdvig = 0
    w = nx.get_node_attributes(g, 'weight')
    result = []
    for i in range(number_of_colors):
        vert[i] = [(x, w[x]) for x in colors if colors[x]==i]
        for v in vert[i]:
            #result.append([i, v[0], sdvig, sdvig+v[1]])
            result.append([int(v[0]), sdvig, sdvig+v[1]])
        sdvig += max(list(map(lambda x: x[-1], vert[i])))
        #print(sdvig)
    right_border = max([x[-1] for x in result])
    return np.asarray(result), right_border



def combinations_(arr):
    # Python3 program to find combinations from n
    # arrays such that one element from each
    # array is present

    # function to print combinations that contain
    # one element from each of the given arrays

    # number of arrays
    n = len(arr)

    # to keep track of next element
    # in each of the n arrays
    indices = [0 for i in range(n)]

    # list of combinations
    combs = []
    #  current combination
    cur_comb = []

    while True:

        for i in range(n):
            cur_comb.append(arr[i][indices[i]])

        combs.append(cur_comb.copy())
        cur_comb.clear()

        # find the rightmost array that has more
        # elements left after the current element
        # in that array
        next = n - 1
        while (next >= 0 and
               (indices[next] + 1 >= len(arr[next]))):
            next -= 1

        # no such array is found so no more
        # combinations left
        if (next < 0):
            return combs

        # if found move to next element in that
        # array
        indices[next] += 1

        # for all arrays to the right of this
        # array current index again points to
        # first element
        for i in range(next + 1, n):
            indices[i] = 0

import itertools
def combinations(arr):
    #print(arr)
    return list(itertools.product(*arr))
    #for elem in itertools.product(arr):
    #    combs.append(elem)
    #return combs

def cycle_check(arr):
    """
    реализация обхода в глубину для поиска циклов
    :param arr: массив, определяющий очередность интервалов
    :return: Boolean, True -- если цикл найден, False -- если не найден

    """
    arr = list(map(int, arr))
    N = len(arr) # кол-во вершин
    paths = []
    for i in range(1, N+1):
        path = [i] # путь из вершины
        prev = arr[i - 1] # предшествующая вершина
        while prev != 0: # условие остановки -- пришли в вершину, которая стартует первой
            if prev in path:
                return True
            path.append(prev)
            prev = arr[prev - 1]
            #paths.append(path)

    # old version

    # for i in range(1, N+1):
    #     if i not in arr: # если вершины нет в массиве, значит онa последняя в очереди
    #         path = [i] # путь из вершины
    #         prev = arr[i - 1] # предшествующая вершина
    #         while prev != 0: # условие остановки -- пришли в вершину, которая стартует первой
    #             if prev in path:
    #                 return True
    #             path.append(prev)
    #             prev = arr[prev - 1]
    #         #paths.append(path)
    #print(paths)
    return False

def get_queue(arr):
    """
    функция, позволяющая восстановить очередность запуска вершин по списку
    пример:
     1 2 3 4 5     [4 -> 3 -> 5 -> 1]
    [0 1 5 3 1] -> [2 -> 1]
    :param arr: массив, определяющий порядок запуска вершин
    :return: очередь, list of lists
    """
    arr = list(map(int, arr))
    N = len(arr)  # кол-во вершин
    paths = []
    for i in range(1, N + 1):
        if i not in arr:  # если вершины нет в массиве, значит онa последняя в очереди
            path = [i]  # путь из вершины
            prev = arr[i - 1]  # предшествующая вершина
            while prev != 0:  # условие остановки -- пришли в вершину, которая стартует первой
                path.append(prev)
                prev = arr[prev - 1]
            paths.append(path)
    #print(paths)
    return paths

def intervals(paths, weights):
    res = [] # структура [[номер вершины, старт, конец], ... []]
    assigned = []
    for p in paths:
        sdvig = 0 # определяет смещение относительно начала
        for i in p[::-1]: # идем в обратном порядке, начиная с самой первой вершины в очереди
            if i not in assigned:
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                res.append([i, sdvig, sdvig+weights[i-1]])
                sdvig += weights[i-1] # добавляем к сдвигу вес предшествующей вершины
                assigned.append(i)
            else:
                sdvig += weights[i-1]

    return sorted(res, key=lambda x: int(x[0]))

# можно попытаться оптимизировать, рассматривая вершины попарно
def contains_banned_intersections(neighb, intervals, N):
   # checked = [] # вершины, проверенные на пересечения с недопустимыми
    print(intervals)
    for i in range(N):
        to_check = [x for x in neighb[i] if x != '0'] # должны проверить соседей и исключить 0
        for r in to_check:
            j=int(r)-1
            points = [intervals[i][-1], intervals[j][-1], intervals[i][1], intervals[j][1]]
            if max(points)-min(points) < ((intervals[i][-1]-intervals[i][1]) + (intervals[j][-1]-intervals[j][1])):
                return True

    return False

# проверка на правильность присвоения интервала для наших алгоритмов (проверяет готовое решение)
def bad_intersections(g, res):
    adj = np.asarray(nx.adjacency_matrix(g).A)
    N = adj.shape[0]  # кол-во вершин
    weights = nx.get_node_attributes(g, 'weight')
    w = []
    for key, value in weights.items():
        w.append(value)  # получаем список весов вершин
    res = sorted(res, key=lambda x: int(x[0]))

    #print(res)
    permut = []
    for i in range(N):
        neighb = list(nx.neighbors(g, str(i + 1)))  # список вершин, смежных для i-й вершины -- могут ей предшествовать
        #neighb.append('0')  # 0 значит, что i-я вершина может не иметь предшественников
        permut.append(list(map(int, neighb)))
    #print(permut)
    for i in range(N):
        for r in permut[i]:
            j = r-1
            points = [res[i][-1], res[j][-1], res[i][1], res[j][1]]
            if max(points)-min(points) < ((res[i][-1]-res[i][1]) + (res[j][-1]-res[j][1])):
                #print(res[i], res[j])
                return True
    return False

def solution(arr):
    # arr -- массив с интервалами для каждой возможной очереди

    rb = []
    for q in arr:
        right_borders = [x[-1] for x in q]
        rb.append(max(right_borders))
    min_ = np.asarray(rb).argmin()


    return np.asarray(arr[min_]), rb[min_]


from tqdm import tqdm

def opt_sol(g):


    adj = np.asarray(nx.adjacency_matrix(g).A)
    N = adj.shape[0] # кол-во вершин

    weights = nx.get_node_attributes(g, 'weight')
    w = []
    for key, value in weights.items():
        w.append(value) # получаем список весов вершин


    permut = [] # матрица перестановок


    # составляем перестановки
    res = 1
    for i in range(N):
        neighb = list(nx.neighbors(g, str(i + 1))) # список вершин, смежных для i-й вершины -- могут ей предшествовать
        neighb.append('0') # 0 значит, что i-я вершина может не иметь предшественников
        permut.append(neighb)
        #permut.append(list(nx.neighbors(g, str(i+1))).append('0')) # добавляем в массив перестановок списки смежных (запрещенных) вершин
        res *= len(permut[i])

    combs = combinations(permut) # все возможные комбинации очередей для интервалов (вершин)

    #!!!!
    #combs = list(combs)
    #print(combs)
    old = len(combs)

    # проверка на циклы
    to_save = []
    for i in tqdm(range(len(combs))):
        if '0' not in combs[i]:
            continue # исключаем варианты, где нет 0 -- хотя бы одна вершина должна начинаться в нуле
        if not cycle_check(combs[i]):
            to_save.append(i)

    combs = [combs[i] for i in to_save]

    to_save.clear()

    interv = []
    for i in tqdm(range(len(combs))):
        interv.append(intervals(get_queue(combs[i]), w))
        if not contains_banned_intersections(permut, interv[i], N):
            to_save.append(i)

    #combs = [combs[i] for i in to_save]
    interv = [interv[i] for i in to_save]

    return solution(interv)

def opt_sol_fast(g):
    adj = np.asarray(nx.adjacency_matrix(g).A)
    N = adj.shape[0]  # кол-во вершин

    weights = nx.get_node_attributes(g, 'weight')
    w = []
    for key, value in weights.items():
        w.append(value)  # получаем список весов вершин

    permut = []  # матрица перестановок

    # составляем перестановки
    res = 1
    for i in range(N):
        neighb = list(nx.neighbors(g, str(i + 1)))  # список вершин, смежных для i-й вершины -- могут ей предшествовать
        neighb.append('0')  # 0 значит, что i-я вершина может не иметь предшественников
        permut.append(neighb)
        # permut.append(list(nx.neighbors(g, str(i+1))).append('0')) # добавляем в массив перестановок списки смежных (запрещенных) вершин
        res *= len(permut[i])
    print(res)

    #interv = []
    to_save = []
    for elem in tqdm(itertools.product(*permut)):
        if '0' not in elem:
            #print('oops')
            continue
        if cycle_check(elem):
            continue

        interv = intervals(get_queue(elem), w)
        #print(elem)
        if not contains_banned_intersections(permut, interv, N):
            to_save.append(interv)
    #print(to_save)
    return solution(to_save)



