import numpy as np
from numpy.random import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

N=100
nodes = [[] for i in range(N)]
A = np.zeros((N,N))
p = float(3) / (N-1)
for n in range(N):
    for l in [i for i in range(N) if i != n]:
        if random() < p:
            nodes[n].append(l)
            A[n,l] = 1
            A[l,n] = 1
# for n in range(N):
#     print(n,nodes[n])
G = nx.from_numpy_array(A)
nx.draw(G,node_size=50)
