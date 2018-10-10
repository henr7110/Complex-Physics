import numpy as np
from numpy.random import random
import networkx as nx
import matplotlib.pyplot as plt

def tris(N,nums,plot=False):
    triangs = 0
    for l in range(nums):
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
        triangs += len([c for c in nx.cycle_basis(G) if len(c)==3])
    return triangs/float(nums)
Ns = [10,100,1000]
triangles = []
for n in Ns:
    triangles.append(tris(n,5))
plt.plot(Ns,triangles,"o")
plt.xlabel("N")
plt.ylabel("triangles")
plt.show()
