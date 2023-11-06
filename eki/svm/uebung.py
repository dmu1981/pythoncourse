import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
# data = np.array([[2.0,4.0],[2.5,2.5],[3.0,5.0],[3.5,2.0],
#                  [4.5,4.5],[5.0,3.5],[3.5,4.0],[4.0,3.5],
#                  [0.5,4.5],[1.0,3.0],[1.5,1.0],[2.0,5.5],
#                  [3.5,5.5],[4.0,0.5],[5.0,1.5],[5.5,5.5],[5.5,3.0]])

# target = [1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

data = np.array([[1.0,1.0],[1.0, 3.0], 
                 [2.0,2.0],[2.5, 1.0],
                 [3.0,3.0],[4.0, 5.0],
                 ])

target = [1,1,-1,-1,1,1]

clf = svm.SVC(kernel='poly', C=1, degree=6)
clf.fit(data,target)
print("Falschklassifizierte")
print(data[clf.predict(data) != target])
print("Support Vektoren")
print(clf.support_vectors_)

x = np.linspace(0, 6, 200)
y = np.linspace(0, 6, 200)
X, Y = np.meshgrid(x, y)
Z = clf.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)
plt.contourf(x, y, Z, cmap=plt.cm.coolwarm, alpha=0.8)

plt.scatter(data.T[0], data.T[1], c=target,
cmap=plt.cm.coolwarm, s=200, edgecolors='k')
plt.plot(clf.support_vectors_.T[0],clf.support_vectors_.T[1], 'k+')

plt.show()