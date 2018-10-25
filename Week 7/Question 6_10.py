# Simulate the collapse model for a N=100 system with γ = 0.00001, assuming that
# company of size s collapse with a probability that decreases with its size as
# ∝ (γ/s)0.2. Simulate time-aggregated size distribution.
# Qlesson: One obtain an exponent for size distribution that is 1/s2, and also
# do this when changing exponent for collapse probability to another number(0.3 or 0.1).
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
N=100
gamma = 0.00001
Omega = 100
def Pcol(s):
    return (gamma/s)**0.2
def Update(s,sum):
    rand = np.random.random()
    if rand < Pcol(s):
        # print("downing")
        return gamma
    else:
        # print("upping")
        return s
n_steps = 200
Ns = np.zeros((N,n_steps))
Ns[:,0] = 1/N
#dsi/dt=Ω·si ·(1−􏰔sj)−ηi(t)·si ,
for t in range(1,n_steps-1):
    sum = np.sum(Ns[:,t-1])
    for i in range(N):
        rand = np.random.randint(N)
        Ns[i,t] = Update(Ns[i,t-1],sum)
    Ns[:,t]= np.copy(Ns[:,t])*Omega
    Ns[:,t] *= 1/np.sum(Ns[:,t])

plt.imshow(Ns,aspect="auto")
plt.colorbar()
plt.xlabel("Time")
plt.ylabel("N")
plt.savefig("Question 6_10 smallplot")
plt.show()
sizes = np.reshape(Ns,n_steps*N)

def f(x, A, B): # this is your 'straight line' y=f(x)
    return B*(x**A)

plt.plot(np.linspace(0,1),f(np.linspace(0,1),-2,11))
plt.hist(sizes,bins=50)
plt.xlabel("size")
plt.yscale("log")
plt.xscale("log")
plt.tight_layout()
plt.show()
