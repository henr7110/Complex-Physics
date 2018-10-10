# Simulate the Bak-Sneppen evolution model for 100 species placed along a line
# in a variant of the model where only one of the neighbors is updated at each
# step. Plot the selected Bmin as function of time, as well as the max of all
# previous selected Bmin’s. How does the minima of B change as time progresses
# toward steady state (look at envelope defined as max over all Bmin at earlier
# times)?
import matplotlib
matplotlib.use('Qt5Agg')
import numpy.random as rdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import curve_fit
N = 100
L = rdm.random(N)

def Update(L):
    i = np.argmin(L)
    L[i] = rdm.random()
    if i == N-1:
        L[0] = rdm.random()
    else:
        L[i+1] = rdm.random()
    return L,L[i]
def animate(i):
    if i % 1 == 0:
        ax1.cla()
        ax2.cla()
        ax1.plot(animate.Bmin)
        ax1.set_title("%d Itteration" % i)
        ax1.set_ylabel(r"$B_{min}$")
        ax1.set_xlabel(r"Itterations")
        ax2.plot(animate.maxBmin)
        ax2.set_xlabel("Itterations")
        ax2.set_ylabel(r"$Max B_{min}$")
        plt.tight_layout()
    animate.L,Bmin = Update(animate.L)
    animate.Bmin.append(Bmin)
    animate.maxBmin.append(max(animate.Bmin))
animate.Bmin = []
animate.maxBmin = []
animate.L = L
fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.plot(animate.Bmin)
ax1.set_title("%d Itteration" % 0)
ax1.set_ylabel(r"$B_{min}$")
ax1.set_xlabel(r"Itterations")
ax2.plot(animate.maxBmin)
ax2.set_xlabel("Itterations")
ax2.set_ylabel(r"$Max B_{min}$")
plt.tight_layout()

anim = animation.FuncAnimation(fig, animate,interval=0)
plt.show()
