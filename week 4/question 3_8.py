import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from numpy.random import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
N=100
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

A = np.zeros((N,N))
p = float(1.5) / (N-1)
for n in range(N):
    for l in [i for i in range(N) if i != n]:
        if random() < p:
            A[n,l] = 1
G = nx.from_numpy_array(A)

nodes = list(range(100))
def Update(G,nodes,plot=False,node_size=10):
    d = len(G.nodes())
    n = np.random.randint(d)
    # print(animate.G.nodes(),animate.nodes)
    # print("removing",nodes[n])
    G.remove_node(nodes[n])
    nodes.remove(nodes[n])
    comps = []
    for i in nodes:
        comps.append(len([c for c in nx.neighbors(G,i)]))
    return G,max(comps),nodes
def animate(i):
    ax1.cla()
    ax2.cla()
    ax2.set_xlabel("Number of nodes removed")
    ax2.set_ylabel("Maximum connectedness")
    nx.draw(G,ax=ax1,node_size=30)
    ax2.plot(range(len(animate.clus)),animate.clus)
    animate.G,c,animate.nodes = Update(animate.G,animate.nodes)
    animate.clus.append(c)
animate.G = G
animate.nodes = nodes
comps = []
for i in range(100):
    comps.append(len([c for c in nx.neighbors(G,i)]))
animate.clus = [max(comps)]
anim = animation.FuncAnimation(fig, animate,interval=2)
plt.show()
