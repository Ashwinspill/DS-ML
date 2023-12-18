import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([4,6,7,8,5,4,3,4,5,7]).reshape(-1,1)
y = np.array([3,4,5,6,7,8,9,3,4,5])

model = LinearRegression()
model.fit(x,y)

slope = model.coef_[0]
inter = model.intercept_

print("Slope", slope)
print("intercept",inter)