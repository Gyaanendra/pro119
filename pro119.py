from numpy.core.fromnumeric import ravel
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import statistics as st
import random as rd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split as tts 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import confusion_matrix as CXM
from sklearn.cluster import KMeans as KM
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz as egx
from six import StringIO 
from IPython.display import Image
import pydotplus


# to change the name of column of a csv file
column_names = ["PassengerId","Pclass","Sex","Age","SibSp","Parch","Survived"]
data_file = pd.read_csv("c119/titanic.csv",names=column_names).iloc[1:]

features =["PassengerId","Pclass","Sex","Age","SibSp","Parch","Survived"]
X = data_file[features]
Y = data_file.Survived
x_train ,x_test ,y_train ,y_test = tts(X,Y,test_size=0.25,random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)

y_predec  =  clf.predict(x_test)
accuracy = AS(y_test,y_predec)

# print(accuracy)
dot_data = StringIO()
egx(clf,out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=features,class_names=["0","1"])

# print(dot_data.getvalue())
graph_data = dot_data.getvalue()
graph = pydotplus.graph_from_dot_data(graph_data)
graph.write_png("diabetes.png")
Image(graph.create_png())