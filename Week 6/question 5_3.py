# # Draw 100000 pairs of random r1(i), r2(i) each from a Gaussian with
# average of and standard deviations 1, and for set define xi(1) =
# −ln(ran1(i))/3, xi(2) = −ln(ran2(i)). Hint: if one draw 12 numbers
# uniformly between 0 1nd 1, their sum is Gaussian with standard deviation 1 and mean 6.
# # Plot histogram of xi(2) − xi(1) for all pairs where xi(2) > xi(1).
# Fit to exponential function.
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
N = 100000
counter = 0
xi = []
while counter < N:
    r1,r2 = np.random.normal(loc=6,scale=1),np.random.normal(loc=6,scale=1)
    while r2 < r1:
        r1,r2 = np.random.normal(loc=6,scale=1),np.random.normal(loc=6,scale=1)
    xi.append(r2-r1)
    counter += 1
def Fitter(M,edges):
    #Fit
    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A*np.exp(B*x)
    A,B = curve_fit(f, edges, M)[0] # your data x, y to fit
    return A,B

hist, bin_edges = np.histogram(xi, bins=100)
plt.hist(xi,bins=100)
A,B = Fitter(hist,bin_edges[:-1])
plt.text(3,3000,"A= %.01f, B= %.01f" %(A,B))
X = bin_edges
Y = A*np.exp(B*X)
plt.plot(X,Y)
plt.savefig("Question 5_3 answer")
plt.show()
