from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
california = datasets.fetch_california_housing()
X = california.data
Y = california.target
print("X")
print(X)
print(X.shape)
print("Y")
print(Y)
print(Y.shape)
l_reg=linear_model.LinearRegression()
plt.scatter(X.T[6],Y)
plt.show()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)#20% data will be used as test and 80% will be used for training
model=l_reg.fit(X_test,Y_test)
predictions=model.predict(X_test)
print("Pridictions: ",predictions)
print("R^2 value: ",l_reg.score(X,Y))
print("coedd: ",l_reg.coef_)
print("intercept: ",l_reg.intercept_)
#powerbi,tableau,scrum,keras