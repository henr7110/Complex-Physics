# simulation of dx/dt=Q-x, Q=10,with x changes in steps of one.
# Then consider simulation where production occurs in steps of 2.
import matplotlib.pyplot as plt
import numpy as np

n_steps = 5000
d1 = 1
Q = 10
X,t = np.zeros(n_steps),np.zeros(n_steps)
X[0],t[0] = 10,0
for i in range(n_steps):
    t_plus = -d1*(1/Q)*np.log(np.random.random())
    t_minus = -d1*(1/X[i-1])*np.log(np.random.random())
    winner = np.argmin([t_plus,t_minus])
    if winner == 0:
        X[i] = X[i-1] + d1
    else:
        X[i] = X[i-1] - d1
    t[i] = t[i-1] + [t_plus,t_minus][winner]
plt.plot(t,X)
l = np.linspace(min(t),max(t))
plt.plot(l,np.full(len(l),np.mean(X)),label="Mean")
plt.plot(l,np.full(len(l),np.sqrt(np.var(X))+np.mean(X)))
plt.plot(l,np.full(len(l),np.mean(X)-np.sqrt(np.var(X))))
plt.text(150,20,"Variance = %.01f" % np.sqrt(np.var(X)))
plt.xlabel("Time")
plt.ylabel("X")
plt.savefig("Question sneppen d1=1")
