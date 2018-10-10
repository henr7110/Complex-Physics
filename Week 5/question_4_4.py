import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import pylab
from matplotlib.cm import ScalarMappable
from matplotlib import animation
import networkx as nx
N = 20
Grid = np.zeros((N,N))
for i in range(N):
    for l in range(len(Grid[i])):
        n = rd.randint(N)
        while n == i:
            n = rd.randint(N)
        Grid[i][l] = n

def Counter(Grid):
    bins = np.zeros((N,N))
    for i in range(N):
        for a in Grid[i]:
            bins[i][int(a)] += 1
    return bins
def Plot(Grid,ax):
    A = np.zeros((N,N))
    for n in range(N):
        taken=np.zeros(N)
        for l in Grid[n]:
            if taken[int(l)] != 1:
                A[n,int(l)] = 1
                taken[int(l)] = 1
    G = nx.from_numpy_array(A)
    nx.draw(G,ax=ax,node_size=50)

def SOC(Grid):
    #Select agent
    agent = rd.randint(N)
    Grid[int(Grid[agent][rd.randint(N)])][rd.randint(N)] = agent
    return Grid
def animate(i):
    if i % 500 ==0:
        ax.cla()
        ax.set_title("interval " + str(i))
        Plot(animate.Grid,ax)
    animate.Grid = SOC(animate.Grid)
animate.Grid = Grid
fig = plt.figure()
ax = plt.subplot()
Plot(animate.Grid,ax)
anim = animation.FuncAnimation(fig, animate,interval=0)
plt.show()
