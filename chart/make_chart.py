import numpy as np
import matplotlib.pyplot as plt

A = np.load('data_A.npy')
B = np.load('data_B.npy')


plt.plot(A[3:,0])
plt.plot(B[3:,0])

plt.xlabel('iteration number')
plt.ylabel('objective value')

plt.show()