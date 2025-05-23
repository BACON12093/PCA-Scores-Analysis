import math
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

A = np.zeros((0,6))
with open("scores", "r") as fp:
    for line in fp:
        fields = np.array(line.split(), dtype = float)
        A = np.vstack((A, fields))

X1 = A[:, 0]
X2 = A[:, 1]
X3 = A[:, 2]
X4 = A[:, 3]
X5 = A[:, 4]
X6 = A[:, 5]

# Step 1
m = 230
b = 1/230
u1 = (b) * sum(X1)
u2 = (b) * sum(X2)
u3 = (b) * sum(X3)
u4 = (b) * sum(X4)
u5 = (b) * sum(X5)
u6 = (b) * sum(X6)
# Step 2
Atil = A - np.array([u1, u2, u3, u4, u5, u6]).reshape(1,-1)
# Step 3
Atmp = Atil/np.sqrt(m-1)
U,s,V = linalg.svd(Atmp,0)
V = V.T
S = np.diag(s)
print(linalg.norm(Atmp-U@S@V.T))
print(V)

g = np.array([0.06, 0.12, 0.12, 0.20, 0.20, 0.30]).reshape(6,1)
gamma = V.T @ g

# Since 1st, 3rd, and 6th row are negative we can just flip those on U and V

Vnew = V * np.array([-1, 1, -1, 1, 1, -1]).reshape(1,-1)
Unew = U * np.array([-1, 1, -1, 1, 1,-1]).reshape(1,-1)

# Then we can just make gamma positive

gammanew = abs(gamma)
print(f"Gamma values: \n{gammanew}")
print(f"Singular Values: \n{s}")
np.set_printoptions(precision = 4)
print(f"Coefficients Expressing Y in terms of X:\n{Vnew}")

B = A @ Vnew
print(B)
Btild = B @ np.diag(gammanew.flatten())
print(Btild)
final = B @ gammanew

# Make 6 scatter plots

plt.figure(figsize=(16, 9))
for i in range(6):
 plt.subplot(2, 3, i + 1)
 plt.scatter(final, Btild[:, i], marker='.')
 plt.title(f"Principal Component {i+1}", fontsize=14)
 plt.xlabel("Score in class", fontsize=13)
 plt.ylabel(rf"$\gamma_{{{i+1}}}Y_{{{i+1}}}$", fontsize=13)
 if i == 0:
    plt.ylim((20, 100))
plt.subplots_adjust(wspace=0.25, hspace=0.35)
plt.show()
