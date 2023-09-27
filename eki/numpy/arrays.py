import numpy as np

# Ein Vektor
a = np.array([1, 2, 3])
print("a: ", a)

# Sein Shape
print("a.shape: ", a.shape)

# Komponentenweise Multiplikation
print("3*a: ", 3*a)

# Komponentenweise Addition
print("3+a: ", 3+a)

# Skalarprodukt
print("a dot b: ", a.dot(np.array([1,1,-1])))

# Eine Matrix
A = np.array([[1,2,1],[2,3,1],[3,6,4]])
print("A:\n", A)

# Transponieren
print("A.T:\n", A.T)

# Matrix Addition
B = np.array([[1,0,1],[1,0,2],[3,0,2]])
print("A+B:\n", A+B)

# Komponentenweise Multiplikation
print("A*B:\n", A*B)

# Matrix-Multiplikation
print("A@B:\n", A@B)

# Matrix-Inverse
print("A @ inverse(A) = ", A @ np.linalg.inv(A))


