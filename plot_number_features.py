import matplotlib.pyplot as plt
import numpy as np


optimal_features=np.loadtxt(open("optimal_features.csv", "rb"), delimiter=",", skiprows=1)

x = np.array(optimal_features[:,0])
y = np.array(optimal_features[:,1])
e = np.array(optimal_features[:,2]/2)


plt.figure()

plt.plot(x,y,color='black')
plt.errorbar(x, y, e, linestyle='None', marker='o', color='red')


plt.xlabel('Number of features')
plt.ylabel('Mean AUC scores')

plt.savefig("%s.pdf" % 'optimal_features')

plt.show()

plt.close()

