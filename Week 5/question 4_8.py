# Consider a system of N = 1000 sites. Assign a random number xi in [0, 1] to
# each species. At each time step select an external noise η from a narrow
# distribution p(η) ∝ exp(−η/σ), σ = 0.1. At each time step: Replace all x < η
# with new random numbers ∈ [0; 1] and, in addition, select one random species j
# and set its xj to a new random number ∈ [0;1]. Simulate this model. Consider
# aftershocks after a large event, say an event where σ > 0.9 and show that they
# obey Omori’s law (Earthquake aftershock frequency decays as 1/time).
import matplotlib
matplotlib.use('Qt5Agg')
import numpy.random as rdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import curve_fit

N = 1000
sigma = 0.1
L = rdm.random(N)
Avalanche = []
tumbler = []
def Fitter(M):
    Y = np.log(np.array([list(M).count(x) for x in set(list(M))]))
    X = np.log(np.array(range(1,len(Y)+1)))
    #Fit
    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A*x + B
    A,B = curve_fit(f, X, Y)[0] # your data x, y to fit
    return A,np.exp(B),np.exp(X),np.exp(Y)
def Update(L,tumbler,a):
    counter = 0
    n = -sigma*np.log(rdm.random())
    if n > 0.9 and a > 1000:
        Shock = True
    else:
        Shock = False
    for i in range(N):
        if L[i] < n:
            L[i] = rdm.random()
            counter += 1
    L[rdm.randint(N)] = rdm.random()
    counter += 1
    if a > 500:
        tumbler.append(counter)
    return np.copy(L),tumbler, Shock
def animate(i):
    if i % 50 == 0 and i >150:
        A,B,X,Y = Fitter(animate.tumbler)
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.hist(animate.L,range=(0,1),bins=100)
        ax1.set_title("%d Itteration" % i)
        ax1.set_ylabel("Counts")
        ax1.set_xlabel(r"$\eta$")
        ax2.plot(animate.tumbler)
        ax2.set_xlabel("Itterations")
        ax2.set_ylabel("Avalanche size")
        animate.title = r'$\tau$=-%.1f, b=%.1f, i = %d' %(A,B,i)
        animate.X = np.linspace(min(X),max(X))
        animate.Y = B*animate.X**A
        ax3.set_title(animate.title)
        ax3.plot(animate.X,animate.Y)
        ax3.hist(animate.tumbler)
        ax3.set_xlabel("Avalanche size")
        ax3.set_ylabel("Count")
    animate.L,animate.tumbler= Update(animate.L,animate.tumbler,i)
Y = np.array([list(tumbler).count(x) for x in set(list(tumbler))])
X = np.array(range(1,len(Y)+1))
animate.X,animate.Y = X,Y
animate.tumbler = tumbler
animate.L = L
fig = plt.figure()
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

ax1.hist(L,range=(0,1),bins=100)
ax1.set_title("%d Itteration" % 0)
ax1.set_ylabel("Counts")
ax1.set_xlabel(r"$\eta$")
ax2.plot(animate.tumbler)
ax2.set_xlabel("Itterations")
ax2.set_ylabel("Avalanche size")
ax3.hist(animate.tumbler)
ax3.set_xlabel("Avalanche size")
ax3.set_ylabel("Count")
plt.tight_layout()
n_steps = 100000
dt = []
cnum = -1
counter = 0
for i in range(n_steps):
    if counter % (n_steps/10) ==0:
        print("%.1f percent done" % (100*counter/n_steps))
    animate.L,animate.tumbler,shock = Update(animate.L,animate.tumbler,counter)
    # if shock:
    #     chocktime = counter
    #     shock = False
    #     cnum += 1
    #     dt.append([])
    #     while not shock:
    #          animate.L,animate.tumbler,shock = Update(animate.L,animate.tumbler,counter)
    #          if animate.tumbler[-1] > 100:
    #              dt[cnum].append(counter - chocktime)
    #          counter += 1

    counter += 1
# bl = []
# for i in dt:
#     for a in i:
#         bl.append(a)
# plt.hist(bl,bins=50)
# plt.yscale("log")
# plt.xscale("log")
# plt.ylabel("Rate of avalanche greater than 5 after quake")
# plt.xlabel("Elapsed time")
# plt.show()
#
A,B,X,Y = Fitter(animate.tumbler)
ax1.cla()
ax2.cla()
ax3.cla()
ax1.hist(animate.L,range=(0,1),bins=100)
ax1.set_title("%d Itteration" % i)
ax1.set_ylabel("Counts")
ax1.set_xlabel(r"$\eta$")
ax2.plot(animate.tumbler)
ax2.set_xlabel("Itterations")
ax2.set_ylabel("Avalanche size")
animate.title = r'$\tau$=-%.1f, b=%.1f, i = %d' %(A,B,i)
animate.X = np.linspace(min(X),max(X))
animate.Y = B*animate.X**A
ax3.set_title(animate.title)
ax3.plot(animate.X,animate.Y)
ax3.hist(animate.tumbler)
ax3.set_xlabel("Avalanche size")
ax3.set_ylabel("Count")
#anim = animation.FuncAnimation(fig, animate,interval=0)
plt.show()
