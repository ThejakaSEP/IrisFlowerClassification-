#Loading Libraris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv('/Users/thejakamahaulpatha/PycharmProjects/IrisFlowerClassification/IRIS.csv')

#Looking at the dimensions of the data set
# print(df.shape)

#Peek at the data
# print(df.head())

#Statistical summary
# print(df.describe())

#Data distrubution by species
# print(df.groupby('species').size())

#Data Visualization

#1. Univariate Plots
    #Creating a boxplot
# df.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
# pyplot.show()
#we can see that there exists some outliers in sepal_width values

    #Creating histogram
# df.hist()
# pyplot.show()

#2. Multivariate Plots
    #Creating a scatter plot
# scatter_matrix(df)
# pyplot.show()

#we can see a diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship


#Create a validaiton Data set ( Splitting the data in to Train and Test)

array = df.values
X = array[:,0:4]
y = array[:,4]

# 80% is for the train and 20% is for the test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=1)

#Build the Models & evaluate
'''
We will try below Algorithms
1) Logistic Regression (LR)  ->  LINEAR ALGO
2) Linear Discriminant Analysis (LDA)   ->  LINEAR ALGO
3) K-Nearest Neighbors (KNN).  -> NON LINEAR ALGO
4) Classification and Regression Trees (CART).  -> NON LINEAR ALGO
5) Gaussian Naive Bayes (NB).  -> NON LINEAR ALGO
6) Support Vector Machines (SVM).  -> NON LINEAR ALGO
'''

#Spot check Algorithms

models = []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))


#evaluate each model in turn
results = []
names = []

for name,model in models:
    kfold = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
    cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)'%(name,cv_results.mean(),cv_results.std()))

# kfold = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
# cv_results = cross_val_score(KNeighborsClassifier(),X_train,y_train,cv=kfold,scoring='accuracy')
# print(cv_results)


#Compare Algorithms

pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparison')
plt.show()
#You can see the SVM (Support Vector Machine ) has the better rate where most of the data is quashed in the upper wisker of the boxplot


#Let's select SVM and Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train,y_train)
predictions = model.predict(X_test)

#Evaluate Predictions
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
