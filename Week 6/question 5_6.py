#Make a Gillespie simulation of the 2-state recruitment model with cooperativity
# as formulated in terms of the different processes in eq. 5.10.
# Let m vary between 0 and 1 in steps of 0.04 and set β = 0.1.
# Qlesson: Should correspond to a agent based model with N = 25.
#(m2(1−m)−m(1−m)2)−β·m+β·(1−m)
import matplotlib.pyplot as plt
import numpy as np

n_steps = 1000
d1 = 0.04
Beta = 0.1
m,t = np.zeros(n_steps),np.zeros(n_steps)
m[0],t[0] = 0.5,0
for i in range(1,n_steps):
    t_plus1 = -d1*(1/(m[i-1]**2*(1-m[i-1])))*np.log(np.random.random())
    t_minus1 = -d1*(1/(m[i-1]*(1-m[i-1])*2))*np.log(np.random.random())
    t_minus2 = -d1*(1/(Beta*m[i-1]))*np.log(np.random.random())
    t_plus2 = -d1*(1/(1-m[i-1]))*np.log(np.random.random())
    winner = np.argmin([t_plus1,t_plus2,t_minus1,t_minus2])
    if winner < 2:
        m[i] = m[i-1] + d1
    else:
        m[i] = m[i-1] - d1
    t[i] = t[i-1] + [t_plus1,t_plus2,t_minus1,t_minus2][winner]
plt.plot(t,m)

l = np.linspace(min(t),max(t))
plt.plot(l,np.full(len(l),np.mean(m)),label="Mean")
plt.plot(l,np.full(len(l),np.sqrt(np.var(m))+np.mean(m)))
plt.plot(l,np.full(len(l),np.mean(m)-np.sqrt(np.var(m))))
# plt.text(150,20,"Variance = %.01f" % np.sqrt(np.var(m)))
plt.xlabel("Time")
plt.ylabel("m")
plt.savefig("Question 5_6 d1=1")
