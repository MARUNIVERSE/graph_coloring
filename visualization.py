from funcs import *
from algos import *
from benchmark_graphs import *
import pandas as pd
from matplotlib import pyplot as plt

i = 10
dens = 0.4
g = graph_constr_mrw(rand_graph_matrix_wv(i, int(dens * i * (i - 1) / 2), seed=1234))
#plot_weighted_graph(g)

#res, t, opt = LF_wv(g)
#res, t, opt = reg_coloring_intervals(g)
res, opt = tree_coloring(g)
#print(res)

print(greedy(g))
#solution_visualization_wv(info=res)
#print(get_graph_info_wv(g))

#print(nx.greedy_color(g))
#print(get_graph_info_wv(g))


