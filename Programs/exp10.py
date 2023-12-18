import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cal = fetch_california_housing()

df = pd.DataFrame(data=cal.data, columns=cal.feature_names)
df['Target'] = cal.target

x = df.drop('Target',axis=1)
y = df['Target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=27)

model = LinearRegression()
model.fit(x_train,y_train)

v = model.predict(x_test)

mean = mean_squared_error(y_test,v)

print(mean)
