from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']

twenty = fetch_20newsgroups(subset='train', categories=categories, shuffle=True,random_state=42)

vector = TfidfVectorizer()
x = vector.fit_transform(twenty.data)
y = twenty.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

sv = SVC(kernel='linear',random_state=42)

sv.fit(x_train,y_train)

v = sv.predict(x_test)

ac = accuracy_score(y_test,v)
print(ac)

cl = classification_report(y_test,v)
print(cl)

new_data = ["this is medicine","This is computer graphics"]

new_x = vector.transform(new_data)
new_v = sv.predict(new_x)

for i,text in enumerate(new_data):
    prediction = twenty.target_names[new_v[i]]
    print(prediction)

