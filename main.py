from funcs import *
from algos import *
from benchmark_graphs import *
import pandas as pd

from time import time

# для игнорирования FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
#   adj = np.array(nx.adjacency_matrix(graph).A)
import warnings
warnings.filterwarnings('ignore')


g = bench_graph(57)
#print(get_graph_info_wv(g))

bench_res = pd.DataFrame(columns=['#', 'Кол-во вершин', 'Кол-во ребер', 'Алгоритм', 'Время, с', 'Правая граница'])

#bench_res = pd.read_excel('Bench_results.xlsx')

strategy = ['largest_first',
            'random_sequential',
            'smallest_last',
            'independent_set',
            'connected_sequential_bfs',
            'connected_sequential',
            'DSATUR']

TEST = 'our_algos' # 'greedy', 'right_border', 'our_algos'

if TEST == 'Huawei':
    for i in range(49, 64+1):
        g = bench_graph(i)
        print(g.number_of_nodes(), g.number_of_edges(), end=' ')
        for j in strategy:
            start = time()
            #res, t, opt = LF_wv(g)
            #res, t, opt = reg_coloring_intervals(g)
            res, opt = greedy(g, strategy=j)
            end = time()-start
            print(f"{i}:", end)
            bench_res = bench_res.append({'#': i,
                              'Кол-во вершин': g.number_of_nodes(),
                              'Кол-во ребер': g.number_of_edges(),
                              'Алгоритм': 'greedy'+' '+j,
                              'Время, с': end,
                              'Правая граница': opt}, ignore_index=True)

err = 0
if TEST == 'greedy':
    test_graph = []
    for dens in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        for i in [10, 50, 100, 150, 1000]:
            test_graph.append(graph_constr_mrw(rand_graph_matrix_wv(i, int(dens * i * (i - 1) / 2), seed=1234)))
    for i, g in enumerate(test_graph, start=1):
        print(i)
        for j in strategy:
            start = time()
            #res, t, opt = LF_wv(g)
            res, opt = sdvig_interv(g, strategy=j)
            #res, opt = tree_coloring(g)
            #opt = opt_sol(g)
            end = time() - start

            if bad_intersections(g, res):
                print('oops')

            bench_res = bench_res.append({'#': i,
                                                  'Кол-во вершин': g.number_of_nodes(),
                                                  'Кол-во ребер': g.number_of_edges(),
                                                  'Алгоритм': 'greedy '+j +' _mod',
                                                  'Время, с': end,
                                                'Правая граница': opt}, ignore_index=True)
    print(err)
if TEST == 'right_border':
    print('aaa')
    test_graph = []
    for dens in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for i in [10, 15]:
            test_graph.append(graph_constr_mrw(rand_graph_matrix_wv(i, int(dens * i * (i - 1) / 2), seed=1234)))
    for i, g in enumerate(test_graph, start=1):
        print(f"{i}, {g.number_of_nodes()}, {g.number_of_edges()}", end=' ')
        start = time()
        res, opt = opt_sol_fast(g)
        end = time() - start
        print(f"правая граница {opt} -- {end}, c")
        bench_res = bench_res.append({'#': i,
                                          'Кол-во вершин': g.number_of_nodes(),
                                          'Кол-во ребер': g.number_of_edges(),
                                          'Время, с': end,
                                          'Правая граница': opt}, ignore_index=True)


if TEST == 'our_algos':
    test_graph = []
    for dens in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        for i in [10, 50, 100, 150, 1000]:
            test_graph.append(graph_constr_mrw(rand_graph_matrix_wv(i, int(dens * i * (i - 1) / 2), seed=1234)))
    for i,  g in enumerate(test_graph, start=1):
        print(i)
        start = time()
        res, opt = LF_sdvig(g)
        end = time() - start

        if bad_intersections(g, res):
            print('oops')

        bench_res = bench_res.append({'#': i,
                                          'Кол-во вершин': g.number_of_nodes(),
                                          'Кол-во ребер': g.number_of_edges(),
                                          'Алгоритм': 'LF_weighted_mod',
                                          'Время, с': end,
                                          'Правая граница': opt}, ignore_index=True)

print(bench_res)
bench_res.to_excel('Bench_results.xlsx')

