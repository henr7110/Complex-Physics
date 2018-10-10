import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from numpy.random import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import curve_fit
#Initdata
n = 1000
M = np.ones(n)
def Fitter(M):
    Y = np.log(np.array([list(M).count(x) for x in set(list(M))]))
    X = np.log(np.array(range(1,len(Y)+1)))
    #Fit
    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A*x + B
    A,B = curve_fit(f, X, Y)[0] # your data x, y to fit
    return A,np.exp(B),np.exp(X),np.exp(Y)

def Update(M):
    #Pick random nodes
    a,b = np.random.randint(n),np.random.randint(n)
    while a == b:
        b = np.random.randint(n)
    #Update
    M[a] = M[a] + M[b]
    M[b] = 1

    return M[:]
def animate(i):
    if i % 20 == 0 and i != 0:
        ax2.clear()
        ax2.hist(M)
        A,B,X,Y = Fitter(animate.M)
        animate.title = r'$\tau$=%.1f, b=%.1f, i = %d' %(A,B,i)
        animate.X,animate.Y = np.linspace(min(X),max(X)),B*animate.X**A
        ax2.plot(animate.X,animate.Y)
        ax2.set_title(animate.title)
        line.set_ydata(M)
        ax1.set_ylim(0,max(M)+2)
    animate.M = Update(M)
#Init plot
animate.title = "Empty"
Y = np.array([list(M).count(x) for x in set(list(M))])
X = np.array(range(1,len(Y)+1))
animate.X,animate.Y = X,Y
animate.M = M
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax2.hist(M)
plt.show()
line, = ax1.plot(M)

#animate
anim = animation.FuncAnimation(fig, animate,interval=0)
plt.show()
