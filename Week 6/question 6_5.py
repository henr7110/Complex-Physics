# Consider the fear factor model with 10 stocks that moves one step up or
# down, all starting at 1000. With probability p = 0.05 all stocks moves
# down simultaneously. What should probability for other up, respective
# down movements in order to let individual stocks perform a unbiased random
# walk? Simulate the system and plot the time series for the average stock price.
# Ups and downs of average is asymmetric, but average change is zero with
# Hurst exponent 1/2.
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
from scipy.optimize import curve_fit


N=10
T_max = 10000
n_steps = T_max * 100
v = np.zeros((n_steps,N))
v[0] = np.full(N,np.log(1000))
p = 0.05
q = 1 / (2*(1-p))
rho = 0.05
cutoff = 1000
def Update(v,n):
    move = False
    if rd.random() < p:
        move = True
    for i in range(N):
        if move:
            v[n,i] = v[n-1,i] - 1
        else:
            if rd.random() < q:
                v[n,i] = v[n-1,i] + 1
            else:
                v[n,i] = v[n-1,i] -1
def Correlate(d,T):
    sum = 0
    points = range(0,len(d),T)
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
def Investor(s,cutoff,rho):
    tau = []
    for i in range(len(s)):
        if i > cutoff:
            break
        #start checker from target
        target = s[i] + s[i]*rho
        n = i
        if s[i] > target:
            done = True
            while s[n] > target:
                n += 1
                if n == len(s)-1 and s[n] > target:
                    done = False
                    break
        else:
            done = True
            while s[n] < target:
                n += 1
                if n == len(s)-1 and s[n] < target:
                    done = False
                    break
        if done:
            tau.append(np.copy(n)-np.copy(i))
    return tau

for n in range(1,n_steps-1):
    if n % (n_steps/10) == 0:
        print("%.01f %% done" % (100*n/n_steps))
    Update(v,n)

v = np.mean(v,axis=1)
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
plt.plot(T,ds,label="s")
plt.plot(T,np.exp(s),label="fit")
plt.text(1,4,"H=%.01f, C=%.01f" %(H,C))
plt.xlabel("log(T)")
# plt.xscale("log")
# plt.yscale("log")
plt.ylabel("s")
plt.title("Time correlated stock info")
plt.legend()
plt.savefig("question 6_5 reg fit")
plt.show()

plt.plot(np.log(T),np.log(ds),label="s")
plt.plot(np.log(T),s,label="fit")
plt.text(1,4,"H=%.01f, C=%.01f" %(H,C))
plt.xlabel("log(T)")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("log(<∆s(T)>)")
plt.title("Time correlated stock info")
plt.legend()
plt.savefig("question 6_5 log fit")
plt.show()


tau_plus = Investor(v,cutoff,rho)
tau_minus = Investor(v,cutoff,-rho)
plus = np.array([[x,float(tau_plus.count(x))] for x in set(tau_plus)])
minus = np.array([[x,float(tau_minus.count(x))] for x in set(tau_minus)])
plus[:,1],minus[:,1] = plus[:,1]/np.sum(plus[:,0]),minus[:,1]/np.sum(minus[:,0])

plt.plot(plus[:,0],plus[:,1],".",label=r"$\tau_+$")
plt.plot(minus[:,0],minus[:,1],".",label=r"$\tau_-$")
plt.legend()
plt.xlabel(r"$\tau$")
plt.ylabel(r"$p(\tau)$")
plt.xscale("log")
plt.savefig("question 6_5 tau")
plt.show()
