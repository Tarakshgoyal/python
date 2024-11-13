from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()
X=iris.data
Y=iris.target
classes=['Iris Setosa','Iris Versicolour','Iris Virginica']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model=svm.SVC()
model.fit(X_train,Y_train)
print(model)
predictions=model.predict(X_test)
acc=accuracy_score(Y_test,predictions)
print("predictions: ",predictions)
print("actual: ",Y_test)
print("accuracy: ",acc)
for i in range(len(predictions)):
    print(classes[predictions[i]])