# Simulate a random walk of uncorrelated up and down movements of s,
# where steps size δ are chosen from the fat-tailed distribution P(δ) ∝ 1/δ3.
# Visualize the walk. Calculate the Hurst exponent by simulation.
# Qlesson: Notice that the mean squared displacement diverge.
# Simulate a walk where a log of a price (s) moves one step up or one step down
# at each time-step. Let the probability to continue in same direction as previous
# step be p = 0.75. Investigate Hurst exponent for this walk numerically. redo
# the simulation for p = 0.99. Hint: Simulate for T up to at least 10000 time-steps,
# for example by following the s-walker from 0 to t =1000000 and average the squared
# of the deviations ∆s(T) of all points separated by T.
#C=⟨−∆s(−T)·∆s(T)⟩t =22H−1−1
# ⟨(∆s(T ))2⟩ = ⟨(s(t + T ) − s(t))2⟩t ∝ T 2H
#
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import numpy.random as rd
import time
def Rand():
    return 1/(np.sqrt(2*rd.random()))
p = 0.5
n_steps = 10000
v,t = np.zeros(n_steps),np.zeros(n_steps)
choice = np.zeros(n_steps)
v[0],t[0] = 10,0
choice[0] = 1

def Update(v,i,choice,t):
    # print(v[i],choice[i])
    if rd.random() < p:
        v[i] = np.copy(v[i-1]) + choice[i-1]
        choice[i] = np.copy(choice[i-1])
    else:
        # print("different")
        v[i] = np.copy(v[i-1]) - choice[i-1]
        choice[i] = -np.copy(choice[i-1])
    t[i] = t[i-1] + Rand()
    # print(v[i])
for i in range(1,n_steps):
    if i % (n_steps/10) ==0:
        print("%.01f %% done" % (100*i/n_steps))
    Update(v,i,choice,t)
len(t)


def Correlate(d,Ti):
    sum = 0
    n = 0
    p = [0]
    t_start = 0
    timer = time.time()
    while n != len(t)-1:
        while sum < Ti:
            if n == len(t)-1:
                break
            sum += t[n]-t_start
            n+=1
        p.append(n)
        t_start = t[n]
        sum = 0
    sum = 0
    points = p
    for i in range(len(points)):
        if i == 0:
            continue
        else:
            sum += (d[points[i]]-d[points[i-1]])**2
    return sum / (len(points)-1)

def Fitter(X,Y,p0=None):
    #Fit
    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A+B*x
    A,B = curve_fit(f, X, Y,p0=p0)[0] # your data x, y to fit
    return A,B
#Plot stock
plt.plot(t,v,label="price")
plt.xlabel("Time")
plt.ylabel("log(Price)")
plt.title("stock time series")
plt.legend()
plt.savefig("question 6_2 stocksim")
plt.show()

Ts = np.linspace(0.001,5)
ds = np.zeros(len(Ts))
for i in range(len(Ts)):
    if i % 10 == 0:
        print(Ts[i])
    ds[i] = Correlate(v,Ts[i])

ds = ds[2:]
T = Ts[2:]
A,B = Fitter(np.log(T),np.log(ds),p0=[np.log(ds[0]),0.7*2])
# ∆s = C*T 2H
# ∆s = ln(C) + 2H ln(T)
# H = B/2, C = exp(A)
H = B/2
C = np.exp(A)

s = A + B* np.log(t[2:])
plt.plot(T,ds,label="<∆s>")
plt.plot(t[2:],np.exp(s),label="fit")
plt.text(10000,10,"H=%.01f, C=%.01f" %(H,C))
plt.xlabel("T")
# plt.xscale("log")
# plt.yscale("log")
plt.ylabel("<∆s(T)>")
plt.title("Time correlated stock info")
plt.legend()
plt.savefig("question 6_2 Hurst reg fit")
plt.show()

plt.plot(np.log(T),np.log(ds),label="<∆s>")
plt.plot(np.log(t[2:]),s,label="fit")
# plt.text(1,4,"H=%.01f, C=%.01f" %(H,C))
plt.xlabel("log(T)")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("log(<∆s(T)>)")
plt.title("Time correlated stock info")
plt.legend()
plt.savefig("question 6_2 Hurst log fit")
plt.show()
