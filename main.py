from funcs import *
from algos import *
from benchmark_graphs import *
import pandas as pd

# для игнорирования FutureWarning: adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
#   adj = np.array(nx.adjacency_matrix(graph).A)
import warnings
warnings.filterwarnings('ignore')


g = bench_graph(57)
#print(get_graph_info_wv(g))

bench_res = pd.DataFrame(columns=['#', 'Кол-во вершин', 'Кол-во ребер', 'Алгоритм', 'Время, с'])

#bench_res = pd.read_excel('Bench_results.xlsx')

Huawei = False
if Huawei is True:
    for i in range(49, 64+1):
        g = bench_graph(i)
        print(g.number_of_nodes(), g.number_of_edges(), end=' ')
        start = time()
        tree_coloring(g)
        end = time()-start
        print(f"{i}:", end)
        bench_res = bench_res.append({'#': i,
                          'Кол-во вершин': g.number_of_nodes(),
                          'Кол-во ребер': g.number_of_edges(),
                          'Алгоритм': 'LF',
                          'Время, с': end}, ignore_index=True)

else:
    test_graph = []
    for dens in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        for i in [10, 50, 100, 150, 1000]:
            test_graph.append(graph_constr_mrw(rand_graph_matrix_wv(i, int(dens * i * (i - 1) / 2), seed=1234)))
    for i, g in enumerate(test_graph, start=1):
        start = time()
        tree_coloring(g)
        end = time() - start
        bench_res = bench_res.append({'#': i,
                                              'Кол-во вершин': g.number_of_nodes(),
                                              'Кол-во ребер': g.number_of_edges(),
                                              'Алгоритм': 'tree_coloring',
                                              'Время, с': end}, ignore_index=True)


print(bench_res)
bench_res.to_excel('Bench_results.xlsx')
#bench_res.to_excel()

