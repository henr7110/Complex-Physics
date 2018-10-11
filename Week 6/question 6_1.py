# Simulate a walk where a log of a price (s) moves one step up or one step down
# at each time-step. Let the probability to continue in same direction as previous
# step be p = 0.75. Investigate Hurst exponent for this walk numerically. redo
# the simulation for p = 0.99. Hint: Simulate for T up to at least 10000 time-steps,
# for example by following the s-walker from 0 to t =1000000 and average the squared
# of the deviations ∆s(T) of all points separated by T.
#C=⟨−∆s(−T)·∆s(T)⟩t =22H−1−1
# ⟨(∆s(T ))2⟩ = ⟨(s(t + T ) − s(t))2⟩t ∝ T 2H
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import numpy.random as rd

p = 0.3
T_max = 10000
n_steps = T_max * 100
v = np.zeros(n_steps)
choice = np.zeros(n_steps)
v[0] = 10
choice[0] = 1

def Correlate(d,T):
    sum = 0
    points = range(0,len(d),T)
    for i in range(len(points)):
        if i == 0:
            continue
        else:
            sum += (d[points[i]]-d[points[i-1]])**2
    return sum / (len(points)-1)

def Update(v,i,choice):
    # print(v[i],choice[i])
    if rd.random() < p:
        v[i] = np.copy(v[i-1]) + choice[i-1]
        choice[i] = np.copy(choice[i-1])
    else:
        # print("different")
        v[i] = np.copy(v[i-1]) - choice[i-1]
        choice[i] = -np.copy(choice[i-1])
    # print(v[i])
for i in range(1,n_steps):
    if i % (n_steps/10) ==0:
        print("%.01f %% done" % (100*i/n_steps))
    Update(v,i,choice)

def Fitter(X,Y,p0=None):
    #Fit
    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A+B*x
    A,B = curve_fit(f, X, Y,p0=p0)[0] # your data x, y to fit
    return A,B
#Plot stock
plt.plot(v,label="price")
plt.xlabel("Time")
plt.ylabel("log(Price)")
plt.title("stock time series")
plt.legend()
plt.savefig("question 6_1 0_3 stocksim")
plt.show()

ds = np.zeros(T_max)

for T in range(2,T_max):
    ds[T] = Correlate(v,T)
ds = ds[2:]
T = range(2,T_max)
A,B = Fitter(np.log(T),np.log(ds),p0=[np.log(ds[0]),0.7*2])
# ∆s = C*T 2H
# ∆s = ln(C) + 2H ln(T)
# H = B/2, C = exp(A)
H = B/2
C = np.exp(A)

s = A + B* np.log(T)
plt.plot(T,ds,label="<∆s>")
plt.plot(T,np.exp(s),label="fit")
plt.text(1,4,"H=%.01f, C=%.01f" %(H,C))
plt.xlabel("log(T)")
# plt.xscale("log")
# plt.yscale("log")
plt.ylabel("log(<∆s(T)>)")
plt.title("Time correlated stock info")
plt.legend()
plt.savefig("question 6_1 p=0_3 reg fit")
plt.show()

plt.plot(np.log(T),np.log(ds),label="<∆s>")
plt.plot(np.log(T),s,label="fit")
plt.text(1,4,"H=%.01f, C=%.01f" %(H,C))
plt.xlabel("log(T)")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("log(<∆s(T)>)")
plt.title("Time correlated stock info")
plt.legend()
plt.savefig("question 6_1 p=0_3 log fit")
plt.show()
