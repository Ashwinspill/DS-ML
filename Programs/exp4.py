from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
kn = KNeighborsClassifier(n_neighbors=7)
kn.fit(x_train,y_train)

v = kn.predict(x_test)
print(v)

cl = classification_report(y_test,v)
print(cl)

ac = accuracy_score(y_test,v)
print(ac)
