import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
A=pd.read_csv("/content/young entrepreneur 1.csv")
A.isnull().sum()
x=A.drop(['y'],axis=1)
y=A['y']
sc=StandardScaler()
X=sc.fit_transform(x)
x_train,x_valid,y_train,y_valid=train_test_split(X,y,test_size=0.3,random_state=0)
y_train.value_counts(normalize=True)
y_valid.value_counts(normalize=True)
x_train.shape,y_train.shape
x_valid.shape,y_valid.shape
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeClassifier(random_state=10)
dt_model.fit(x_train , y_train)
dt_model.score(x_train , y_train)
dt_model.score(x_valid , y_valid)
dt_model.predict(x_valid)
dt_model.predict_proba(x_valid)
y_pred = dt_model.predict_proba(x_valid)[:,1]
new_y = []
for i in range(len(y_pred)):
  if y_pred[i]<0.6:
    new_y.append(0)
  else:
    new_y.append(1)
from sklearn.metrics import accuracy_score
accuracy_score(y_valid,new_y)
train_accuracy=[]
validation_accuracy=[]
for depth in range(1,10):
  dt_model = DecisionTreeClassifier(max_depth=depth , random_state=10)
  dt_model.fit(x_train,y_train)
  train_accuracy.append(dt_model.score(x_train,y_train))
  validation_accuracy.append(dt_model.score(x_valid,y_valid))
frame = pd.DataFrame({'max_depth':range(1,10),'train_acc':train_accuracy,'valid_acc':validation_accuracy})
frame.head()
plt.figure(figsize=(12,6))
plt.plot(frame['max_depth'],frame['train_acc'],marker='o')
plt.plot(frame['max_depth'],frame['valid_acc'],marker='o')
plt.xlabel('Depth of Tree')
plt.ylabel('Performance')
plt.legend()
dt_model = DecisionTreeClassifier(max_depth=8,max_leaf_nodes=25,random_state=10)
dt_model.fit(x_train,y_train)
dt_model.score(x_train,y_train)
dt_model.score(x_valid,y_valid)
from sklearn import tree
!pip install graphviz
decision_tree=tree.export_graphviz(dt_model,out_file='tree.dot',feature_names=x_train.base,max_depth=3,filled=True)
!dot -Tpng tree.dot -o tree.png
image = plt.imread('tree.png')
plt.figure(figsize=(15,15))
plt.imshow(image)
