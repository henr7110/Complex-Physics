# Draw 100000 random numbers ran(i) uniformly between 0 and 1, and for each
# number set xi = −ln(ran(i)). Plot histogram of xi. Fit to exponential function.
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

xi = -np.log(np.random.random(100000))
def Fitter(M,edges):
    #Fit
    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A*np.exp(B*x)
    A,B = curve_fit(f, edges, M)[0] # your data x, y to fit
    return A,B


hist, bin_edges = np.histogram(xi, bins=100)
plt.hist(xi,bins=100)
A,B = Fitter(hist,bin_edges[:-1])
plt.text(5,8000,"A= %.01f, B= %.01f" %(A,B))
X = bin_edges
Y = A*np.exp(B*X)
plt.plot(X,Y)
plt.show()
