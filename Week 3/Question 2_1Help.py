import matplotlib.pyplot as plt
from scipy.ndimage import measurements
import numpy as np
from pylab import shuffle
from pylab import rand
def ClusterCounter(Grid,L,num):
    l = np.zeros(num)
    for i in range(L):
        for j in range(L):
            l[Grid[i,j]-1] += 1
    return l

#core program
L = 400
r = rand(L,L)
p = 0.59275
z = r < p
lw, num = measurements.label(z)
c = np.arange(lw.max() + 1)

# shuffle(c); # (optional)
count = ClusterCounter(lw,L,num)
#plotting tools
#plot size distribution
s = np.linspace(0,100)
plt.plot(s,2500*1/(s**(187/91)))
plt.hist(count,bins=50,range=(0,100))
plt.show()

#plot field
a = np.copy(c)
shuffle(a)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
im = ax.imshow(c[lw], cmap=plt.cm.jet, interpolation='none')
ax.set_axis_off()
plt.show()
