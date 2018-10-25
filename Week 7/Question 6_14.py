# Simulate the quit-and-double game from the for a society with 1000 agents.
# What is survival time distribution? Hint: One could equivalently simulate
# one agent for many time-steps where each collapse set agent to size 1.
# Then a sample of his fortune over a long sequence of situations with be
# identical to the 1000 agents, be- cause they are anyway non-interacting.
# Qlesson: 2−t. Very short, lifetime.
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
def Update(V,Omega,p):
    """Simulates roulette guy...."""
    rand = np.random.random()
    if rand < p:
        return 1
    else:
        return V*Omega

#Simulation for catastrophic events
n_steps = 1000000
Omega = 2
p = 0.5
N_0 = 1
results = []
Ks = np.zeros(n_steps)
Ks[0] = N_0
for i in range(1,n_steps):
    Ks[i] = Update(Ks[i-1],Omega,p)
plt.plot(Ks)
plt.title("Time series for quit or double game")
plt.yscale("log")
plt.show()

Y = np.array([list(Ks).count(x) for x in set(list(Ks))])
X = pd.unique(Ks)
def f(x, A, B): # this is your 'straight line' y=f(x)
    return B*(x**A)
A,B = curve_fit(f, X, Y)[0] # your data x, y to fit
plt.plot(X,f(X,A,B),label="fit")
plt.plot(X,Y,label="data")
plt.legend()
plt.text(1e3,1e5,"B*x^(A)   A=%.01f,B=%.01f"%(A,B))
plt.xlabel("Fortune")
plt.ylabel("Count")
plt.yscale("log")
plt.xscale("log")
plt.savefig("Answer question 6_14 fortune dist")
plt.show()

ds = []
start = 0
for i in range(1,len(Ks)):
    if Ks[i] == 1:
        ds.append(i-start)
        start=i
plt.hist(ds,label="data")
def f(x, A, B): # this is your 'straight line' y=f(x)
    return B*(A**(-x))
Y = np.array([list(ds).count(x) for x in set(list(ds))])
X = np.sort(pd.unique(ds))
A,B = curve_fit(f, X, Y)[0] # your data x, y to fit
plt.plot(X,f(X,A,B),label="fit")
plt.text(10,200000,"B*(A**(-t))    A=%.01f, B=%.01f" % (A,B))
plt.legend()
plt.xlabel("survival time")
plt.ylabel("count")
plt.savefig("Quesiton  6_14 Survival time ")
plt.show()
