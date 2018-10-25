# Simulate the long time (500 updates) development of a capital that grows with
# rate Ω = 2 during good times, but is exposed to catastrophic events with prob-
# ability p = 0.1. In case of these events they loose everything invested.
# Simulate the development of initial capital of 1 when using the Kelly optimum
# value of x. Also simulate the development with other values of x, e.g. x = 0.01
# and x = 0.9 and compare outcomes. Repeat simulation with finite disasters,
# say that bad events leads to reduction of invested fortune with a factor
# ω = 10−2, respectively ω = 0.5. Hint: simulate the development of the log
# of the capital (where each event amount to addition or subtraction of the
# log of the change).
# Qlesson: There is an optimum, but the gain with varying around that optimum is quite soft.

import numpy as np
import matplotlib.pyplot as plt
def Update(V,Omega,p,w):
    """Simulates roulette guy...."""
    rand = np.random.random()
    if rand < p:
        return V*w
    else:
        return V*Omega

#Simulation for catastrophic events
n_steps = 500
Omega = 2
w = 0
p = 0.1
N_0 = 1
xK = float(p)*(Omega/(Omega-1)) #Kelly optimum
xs = [xK,0.01,0.001,0.90]

results = []
for x in xs:
    N=1000
    sum = np.zeros(n_steps)
    for n in range(N):
        Ks = np.zeros(n_steps)
        Ks[0] = N_0
        for i in range(1,n_steps):
            F = Ks[i-1]*x
            Ks[i] = F + Update(Ks[i-1]-F,Omega,p,w)
        sum += Ks
    sum = sum/N
    results.append(np.copy(sum))
min(results[2])
for i,x in zip(results,xs):
    plt.plot(i,label="x = %.03f"%x)
plt.legend()
plt.yscale("log")
plt.savefig("Results Question 6_9")
plt.show()
