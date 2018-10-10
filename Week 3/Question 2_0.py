import matplotlib.pyplot as plt
import numpy as np

tau = [1,1.5,2,2.3,3]
s = np.array([1,2,3,])
for t in tau:
    plt.plot(s,1/(s**t),"o",label="tau = %.01f"%t)
plt.legend()
plt.show()
