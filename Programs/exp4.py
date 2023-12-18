import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sea

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
v=knn.predict(x_test)
accuracy_score=accuracy_score(y_test,v)
print(accuracy_score)
classification_report=classification_report(y_test,v)
print(classification_report)
confusion_matrix=confusion_matrix(y_test,v)
print(confusion_matrix)

plt.figure(figsize=(10,8))
sea.heatmap(confusion_matrix,annot=True,cmap='Blues',fmt='g')
plt.show()


# new_data = np.array([5.1, 3.5, 1.4, 0.2]).reshape(1,-1)
# predic=knn.predict(new_data)
#
# predicted=iris.target_names[predic[0]]
# print(predicted)