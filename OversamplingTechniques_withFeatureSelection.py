import numpy as np
import pandas as pd
X = np.load('data_withFeatureSelection.npy')
y = np.load('label.npy')

###########Random Oversampling#############
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
auc_mean=0
conf_mat_micro = np.zeros([2,2])
for i in range(10):
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    conf_mat = np.zeros([2,2])
    auc = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = SVC(kernel='linear', C=1)
        oversample = RandomOverSampler()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        conf_mat += confusion_matrix(y_test,pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        auc += metrics.auc(fpr, tpr)
    print(auc/5)
    auc_mean += auc/5
    print(conf_mat)
    conf_mat_micro += conf_mat
print('Results:')
print('Average AUC:',auc_mean/10)
print('Micro confusion matrix:',conf_mat_micro)

###########SMOTE#############
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
auc_mean=0
conf_mat_micro = np.zeros([2,2])
for i in range(10):
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    conf_mat = np.zeros([2,2])
    auc = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = SVC(kernel='linear', C=1)
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        conf_mat += confusion_matrix(y_test,pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        auc += metrics.auc(fpr, tpr)
    print(auc/5)
    auc_mean += auc/5
    print(conf_mat)
    conf_mat_micro += conf_mat
print('Results:')
print('Average AUC:',auc_mean/10)
print('Micro confusion matrix:',conf_mat_micro)

###########Borderline SMOTE#############
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.svm import SVC

from imblearn.over_sampling import BorderlineSMOTE
auc_mean=0
conf_mat_micro = np.zeros([2,2])
for i in range(10):
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    conf_mat = np.zeros([2,2])
    auc = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = SVC(kernel='linear', C=1)
        oversample = BorderlineSMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        conf_mat += confusion_matrix(y_test,pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        auc += metrics.auc(fpr, tpr)
    print(auc/5)
    auc_mean += auc/5
    print(conf_mat)
    conf_mat_micro += conf_mat
print('Results:')
print(auc_mean/10)
print(conf_mat_micro)

###########ADASYN#############
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn import metrics

from imblearn.over_sampling import ADASYN
auc_mean=0
conf_mat_micro = np.zeros([2,2])
for i in range(10):
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    conf_mat = np.zeros([2,2])
    auc = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = SVC(kernel='linear', C=1)
        oversample = ADASYN()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        conf_mat += confusion_matrix(y_test,pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        auc += metrics.auc(fpr, tpr)
    print(auc/5)
    auc_mean += auc/5
    print(conf_mat)
    conf_mat_micro += conf_mat
print('Results:')
print('Average AUC:',auc_mean/10)
print('Micro confusion matrix:',conf_mat_micro)