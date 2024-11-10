# Importing the libraries
import pandas as pd
import numpy as np

#load the dataset
data = pd.read_csv('salary.csv')

# print(data.head())
# print(data.shape) # number of rows and columns (32561 ,5)

# Mapping data  to binary values
income_set = set(data['income'])
data['income'] = data['income'].map({'<=50K':0, '>50K':1}).astype(int)
# print(data.head(10))
 
#Segregating the data into training and testing
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

#Splitting the data into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Finding the nearest neighbors best for the salary estimation
error = []
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Claculating  error for K values between 1 and 40
for k in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
print(error)    
plt.figure(figsize=(12,6))
plt.plot(range(1,40),error ,color='red' , linestyle='dashed' , marker='o',markersize=10)
plt.title(' Mean Error vs K value')
plt.xlabel('K value')
plt.ylabel(' Mean Error')
plt.show()

# modle Training 
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=2 , metric='minkowski', p=2)
model_knn.fit(X_train, y_train)

#model evaluation
y_pred = model_knn.predict(X_test)
print(y_pred)


from sklearn.metrics import accuracy_score , confusion_matrix
print(accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))

age=int(input('Enter your age: '))
edu=int(input('Enter your education: '))
cg=int(input('Enter your captial gain: '))
wh=int(input('Enter your Hours Per week: '))
newEmp=[[age,edu,cg,wh]]

result=model_knn.predict(newEmp)
print(result)

if result ==1:
    print('Salary is high')
else:  
    print('Salary is low')


 #Finding the nearest neighbors best for the salary estimation
accuracy = []
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#Claculating  error for K values between 1 and 40
for k in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    accuracy.append(accuracy_score(pred_i, y_test).__format__('.3f'))
print(accuracy)
plt.figure(figsize=(12,6))
plt.plot(range(1,40),accuracy ,color='red' , linestyle='dashed' , marker='o',markersize=10)
plt.title(' Mean Error vs K value')
plt.xlabel('K value')
plt.ylabel(' accuracy')
plt.show()
## model Training
## model Training
###DDD
