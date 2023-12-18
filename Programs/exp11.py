from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=27)

dec = DecisionTreeClassifier(max_depth=3)
dec.fit(x_train,y_train)
v = dec.predict(x_test)

cl = classification_report(y_test,v)
print(cl)

ac = accuracy_score(y_test,v)
print(ac)

plt.figure(figsize=(10,15))
plot_tree(dec,filled=True,feature_names=iris.feature_names , class_names=iris.target_names)
plt.title("tree")
plt.show()