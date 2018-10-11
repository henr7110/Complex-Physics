# Plot eq. 6.5 as function of hurst exponent H, and interpret this in terms of
# profit of a sensible strategy. Devise an investment strategy and calculate
# the maxi- mum average profit per investment step for a H = 0.4 market.
# C=⟨−∆s(−T)·∆s(T)⟩t =22H−1−1
import matplotlib.pyplot as plt
import numpy as np

H = np.linspace(0,1)
C = 2**(2*H-1)-1
C1 = np.zeros(len(H))
plt.plot(H,C)
plt.plot(H,C1)
plt.xlabel("H")
plt.ylabel("C")
plt.show()
#Strategy: Buy at t if s(t−T) > s(t) (6.7) Sell at t if s(t−T) < s(t)
Avg_corr = 2**(2*0.4-1)-1
Avg_corr
P_to_follow_trend = 4**(0.4-1)
P_to_follow_trend
P_to_not_follow_trend = 1-P_to_follow_trend
P_to_not_follow_trend
