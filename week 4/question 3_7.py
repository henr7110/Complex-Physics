import numpy as np
from numpy.random import random
import networkx as nx
import random as rd
import matplotlib.pyplot as plt
#Modular network
trimod = []
trirand = []
for i in range(30):
    N=10
    A = np.zeros((N**2,N**2))
    p_in = 0.01
    p_same = 0.5
    colors = [b for b in range(N) for i in range(N)]
    x,y = np.linspace(-0.4,0.4,10),np.linspace(-0.4,0.4,10)
    rd.shuffle(x)
    rd.shuffle(y)
    pos = {}
    groupcens = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1)]
    counter = 0
    for i in range(N**2):
        if i % 10 == 0 and i != 0:
            counter += 1
        pos[i] = (groupcens[counter][0] + x[counter],groupcens[counter][1] + y[counter])
        rd.shuffle(x)
        rd.shuffle(y)

    for i in range(N):
        for n in range((i)*N,(i+1)*N):
            for l in [i for i in range((i)*N,(i+1)*N) if i != n]:
                if random() < p_same:
                    A[n,l] = 1
                    A[l,n] = 1
            if random() < p_in:
                x,y = [a for a in range(100) if a not in range((i)*N,(i+1)*N)][np.random.randint(90)],[a for a in range(100) if a not in range((i)*N,(i+1)*N)][np.random.randint(90)]
                A[x,y] = 1
                A[y,x] = 1
    G = nx.from_numpy_array(A)
    trimod.append(len([c for c in nx.cycle_basis(G) if len(c)==3]))
    #nx.draw(G,pos=pos,node_color=colors,node_size=30,width=0.5)
    #Random network
    A1 = np.zeros((100,100))
    for n in range(100):
        for i in range(6):
            l = [i for i in range(100) if i != n][np.random.randint(99)]
            if random() < p_same:
                A1[n,l] = 1
                A1[l,n] = 1
    G1 = nx.from_numpy_array(A1)
    trirand.append(len([c for c in nx.cycle_basis(G1) if len(c)==3]))
plt.hist(trimod,label="modular")
plt.hist(trirand,label="random")
plt.xlabel("Number of triangles")
plt.ylabel("Count")
plt.legend()
plt.show()
