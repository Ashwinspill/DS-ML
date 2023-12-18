import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

load = load_breast_cancer()

x = load.data
y = load.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=27)

dec = DecisionTreeClassifier(max_depth=4)
dec.fit(x_train,y_train)

v = dec.predict(x_test)

accuracy_score = accuracy_score(y_test,v)
print(accuracy_score)
cl = classification_report(y_test,v)
print(cl)

plt.figure(figsize=(10,15))
plot_tree(dec,filled=True,feature_names=load.feature_names, class_names=load.target_names)
plt.title("ahhaha")
plt.show()