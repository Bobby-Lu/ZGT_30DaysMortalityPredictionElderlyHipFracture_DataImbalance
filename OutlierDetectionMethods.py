import numpy as np
X = np.load('data.npy')
y = np.load('label.npy')

#########One-class SVM###########
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn import metrics
kf = KFold(n_splits=5, random_state=None, shuffle=False)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
conf_mat = np.zeros([2,2])
auc = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = OneClassSVM(gamma='scale', kernel='rbf', nu=0.08)
    X_train = X_train[y_train==0]
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    pred[pred==1]=0
    pred[pred==-1]=1
    conf_mat += confusion_matrix(y_test,pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    auc += metrics.auc(fpr, tpr)
print(conf_mat)
print(auc/5)

#########Isolation Forest###########
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=None, shuffle=False)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
conf_mat = np.zeros([2,2])
auc = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = IsolationForest(contamination=0.08,n_estimators=58)
    X_train = X_train[y_train==0]
    model.fit(X_train)
    pred = model.predict(X_test)
    pred[pred==1]=0
    pred[pred==-1]=1
    conf_mat += confusion_matrix(y_test,pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    auc += metrics.auc(fpr, tpr)
print(conf_mat)
print(auc/5)

#########Local Outlier Factor###########
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix
from numpy import vstack
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=None, shuffle=False)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
conf_mat = np.zeros([2,2])
auc = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LocalOutlierFactor(contamination=0.08)
    X_train = X_train[y_train==0]
    composite = vstack((X_train,X_test))
    pred = model.fit_predict(composite)
    pred[pred==1]=0
    pred[pred==-1]=1
    pred = pred[len(X_train):]
    conf_mat += confusion_matrix(y_test,pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    auc += metrics.auc(fpr, tpr)
print(conf_mat)
print(auc/5)

#########DB SCAN###########
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
conf_mat = np.zeros([2,2])
model = DBSCAN(eps=35,min_samples=50)
pred = model.fit(X).labels_
pred[pred!=-1]=0
pred[pred==-1]=1
conf_mat += confusion_matrix(y,pred)
fpr, tpr, thresholds = metrics.roc_curve(y, pred)
print(conf_mat)
print(metrics.auc(fpr, tpr))