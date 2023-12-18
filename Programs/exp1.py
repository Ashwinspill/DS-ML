import numpy as np

rows = int(input("the rows num :"))
columns = int(input("coloumns :"))

matrix1 = []

for i in range(rows):
    a = []
    for j in range(columns):
        a.append(int(input()))
    matrix1.append(a)

for i in range(rows):
    for j in range(columns):
        print(matrix1[i][j], end="")
    print()

rows1 = int(input("the rows num :"))
columns1 = int(input("coloumns :"))

matrix2 = []

for i in range(rows):
    a = []
    for j in range(columns):
        a.append(int(input()))
    matrix2.append(a)

for i in range(rows):
    for j in range(columns):
        print(matrix2[i][j], end="")
    print()


sum = np.add(matrix1,matrix2)
print(sum)