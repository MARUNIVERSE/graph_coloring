from funcs import *
from algos import *
from benchmark_graphs import *
import pandas as pd
from matplotlib import pyplot as plt

i = 10
dens = 0.3
g = graph_constr_mrw(rand_graph_matrix_wv(i, int(dens * i * (i - 1) / 2), seed=1234))

#%%

#plot_weighted_graph_wv(g)

#res, t, opt = LF_wv(g)
#res, t, opt = reg_coloring_intervals(g)
#res, opt = tree_coloring(g)
#print(res)

#print(greedy(g))
#solution_visualization_wv(info=res)
#print(get_graph_info_wv(g))

#print(nx.greedy_color(g, strategy='largest_first'))
#print(get_graph_info_wv(g))

#res, rb = greedy(g, strategy='largest_first')
#solution_visualization_wv(info=res)

#print(greedy(g))
#print(nx.greedy_color(g))
#print(get_graph_info_wv(g)[1])


#print(get_queue([0,1,5,3,1]))

#res, rb = opt_sol(g)
#print(res)
#solution_visualization_wv(info=res)
#print(rb)
#


#print(intervals(get_queue([0,1,5,3,1]), [3,2,1,1,2]))
#g = graph_constr_mrw(rand_graph_matrix_wv(10, 13, seed=1234))
#res, rb = greedy_mod(g, strategy='random_sequential')
#print(res, rb)
#solution_visualization_wv(info=res)


#%%
i = 10
dens = 0.1
g = graph_constr_mrw(rand_graph_matrix_wv(i, int(dens * i * (i - 1) / 2), seed=1234))

#PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

print('aa')
res, rb = opt_sol_fast(g)
print(res, rb)

#res, rb = tree_coloring_mod(g)
#print(rb)
