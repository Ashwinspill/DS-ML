import numpy as np

matrix = np.array([[3,4,5],
                   [3,6,7],
                   [7,2,1]])

U,S,VT = np.linalg.svd(matrix)

print("U")
print(U)
print("S")
print(np.diag(S))
print("VT")
print(VT)

recon = np.dot(U,np.dot(np.diag(S),VT))
print("R")
print(recon)