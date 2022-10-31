def LF_wv(graph):
    nodes_dict = nx.get_node_attributes(graph, 'weight')  # получаем инфу в виде словаря
    adj = np.array(nx.adjacency_matrix(graph).A)
    if adj.shape[0] != len(nodes_dict.keys()):
        print("smth bad happened, idk")
    # делаем из словаря список
    nodes = []
    for key, value in nodes_dict.items():
        nodes.append((int(key), value, np.sum(adj[::,int(key)])))
    nodes = np.array(nodes, dtype=[('node', int),('weight', int), ('degree', int)])

    start = time()

    sorted_graph = np.sort(nodes, order=['weight', 'degree'])[::-1]
    # print(sorted_graph)
    # earliest_time = 0
    N = len(nodes)
    res = np.zeros((N, 3))
    for i in range(N):
        idx = sorted_graph[i]['node']
        if i ==0:
            res[idx][0]=idx
            res[idx][1]=0
            res[idx][2]=sorted_graph[i]['weight']
            continue
        res[idx][0] = idx
        earliest = 0 # время старта для i вершины
        adj_v = [] #--
        for t in range(i):
            if adj[idx, sorted_graph[t]['node']] > 0:
                adj_v.append(tuple(res[sorted_graph[t]['node']]))  
        adj_v = np.array(adj_v,dtype=[('node', int),('start', int), ('end', int)])
        adj_v = np.sort(adj_v,order=['start'])
        #print(adj_v,'to',sorted_graph[i])
        for p in adj_v:
            if earliest + sorted_graph[i]['weight'] <= p['start']: # если в рассмотренном множестве мы нашли смежную с нашей вершину
                break
            else:
                earliest = max(earliest, p['end'])
                #print('earliest updated to',earliest,'by',p)
        res[idx][1] = earliest
        res[idx][2] = earliest + sorted_graph[i][1]
    end = time()
    opt = max(res[::, 2])
    return res, end - start, opt

# Последняя версия для LF
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
  
  #### CLIQUE SAMPLING
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
    choose = set(range(N))
    while len(Startset) < samples and len(Startset)<N: # выбираем вершины для построения подграфов
        Startset.add(random.choice(tuple(choose)))
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
    Notincli = np.sort(Notincli,order=['weight','degree'])[::-1]
    
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
    answer = np.zeros((N,3))
    left = 0
    iteration = 0
    for i in range(len(initColoring)):
        for j in initColoring[i]:
            answer[iteration,::]=np.array([j['node'],left,left+j['weight']])
            iteration+=1
        left+=np.max(np.array(initColoring[i],
                              dtype=[('node', int),('weight', int), ('degree', int),('color',int)])['weight'])
    return answer[:iteration], left
  
  #### CLIQUE SAMPLING MOD
  def Clique_mod(g, samples=10): #samples - число вершин, из которых будут строиться полные подграфы
    
    coloring,_=Clique_sampling(g,samples=samples)
    
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
