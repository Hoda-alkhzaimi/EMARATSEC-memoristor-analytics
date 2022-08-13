
import numpy as np
import pandas as pd
import os
from scipy import io
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb   


# Prepare the data
os.chdir("../../data/")
set9 = io.loadmat('set9.mat')
set10 = io.loadmat('set10.mat')
set11 = io.loadmat('set11.mat')
set17 = io.loadmat('set17.mat')
set18 = io.loadmat('set18.mat')

set25 = io.loadmat('set25.mat')
set26 = io.loadmat('set26.mat')

set27 = io.loadmat('set27.mat')
set28 = io.loadmat('set28.mat')
set33 = io.loadmat('set33.mat')

voltage = np.concatenate((set9['V'].reshape(128), set10['V'].reshape(128), set11['V'].reshape(128), set17['V'].reshape(128), set18['V'].reshape(128), set25['V'].reshape(128),  set26['V'].reshape(128), set27['V'].reshape(128), set28['V'].reshape(128), set33['V'].reshape(128)))
current = np.concatenate((set9['C'].reshape(128), set10['C'].reshape(128), set11['C'].reshape(128), set17['C'].reshape(128), set18['C'].reshape(128), set25['C'].reshape(128),  set26['C'].reshape(128), set27['C'].reshape(128), set28['C'].reshape(128), set33['C'].reshape(128)))

set9str = "11010010110011001001010001101010101110111010100101000000010011100110110001011101010010010000001111000011001010010011000000000111"
set10str = "00100010010000101000111010100000100100100010000000001010100010011111111100001111000100000110100100100001001000110000100010000100"
set11str = "11010010110011001001010001101010101110111010100101000000010011100110110001011101010010010000001111000011001010010011000000000111"
set17str = "11110010111010011111101000001011110010101000000000001000010110000001111011110001011010111111011100101101100111001110101000101001"
set18str = "11001100110011001001010011010100100001000001101000010111010011001010000010001111010111101010001111100011010000100000110000000100"

set25str = "10101001010001001001011101110000000000011001000000110000010000010011001010101001101011001111000010110110100000000100001110101111"
set26str = "01011101011101000000000000000010010010111100001001101001000101101110000101100010101011000010001010101101101001100000111000011000"
set27str = "10010101111001110011101111101110100011011111110010000110110101111100000001111100010010010101010000101100001001001000011001110101"
set28str = "10110111001111011001010011100000100000100010011011011110011011100110100110010100001110101101011011000100101010100001000110000010"
set33str = "01111100000101001110111111110001010100101010110000000001011010001000000000011101000110000111100010010111100010100001010100010101"

def binstr(string):
    """Splits a string into a list of binary digits

    Args:
        string : binary values as string

    Returns:
        array(int) : binary values as array of int  
    """
    actual_bin = []
    for i in range(len(string)):
        if string[i] == '1':
            actual_bin.append(1)
        else:
            actual_bin.append(0)
    return actual_bin



binary = np.concatenate((binstr(set9str), binstr(set10str), binstr(set11str), binstr(set17str), binstr(set18str), binstr(set25str),binstr(set26str), binstr(set27str), binstr(set28str), binstr(set33str)))
X = np.column_stack((voltage, current)) # combine all the values into one array
y = binary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12345)



print(f"Shape of X-train: {X_train.shape})")


print(f"Shape of X-test: {X_test.shape})")

clf = LogisticRegression(random_state=0).fit(X_train, y_train) # logistic regression
predictions = clf.predict(X_test)
lr_accuracy = accuracy_score(predictions, y_test)
lr_roc = roc_auc_score(y_test, predictions)
print("----------------------------------------------------")
print(f"Logistic regression accuracy: {lr_accuracy}")
print(f"Logistic regression roc: {lr_roc}")
print("----------------------------------------------------")


# Kmeans

X_cluster = np.column_stack((voltage, current)) # reshape data for clustering

kmeansP = KMeans(n_clusters=2, random_state=12454,verbose=0, init="k-means++").fit(X_train) # Kmeans++
kmeans = KMeans(n_clusters=2, random_state=0,init='random').fit(X_train) # Kmeans

predskp = kmeansP.predict(X_test)
predsk = kmeans.predict(X_test)

predskp = kmeansP.predict(X_test)
predsk = kmeans.predict(X_test)


kmeans_accuracy = accuracy_score(predsk, y_test)
kmeansP_accuracy = accuracy_score(predskp, y_test)
kmeans_roc =  roc_auc_score(y_test, predsk)
kmeansP_roc = roc_auc_score(y_test, predskp)

print(f"Kmeans accuracy: {kmeans_accuracy}; ROC: {kmeansP_roc}")
print(f"Kmeans++ accuracy: {kmeansP_accuracy}; ROC: {kmeansP_roc}")
print("----------------------------------------------------")

# Gaussian Mixture Model

gm = GaussianMixture(n_components=2, random_state=0).fit(X_train)

gm_accuracy = accuracy_score(gm.predict(X_test), y_test)
gm_roc = roc_auc_score(y_test, gm.predict(X_test))
print(f'GMM accuracy: {gm_accuracy}; ROC: {gm_roc}')
print("----------------------------------------------------")

# SVM

svm = svm.SVC(kernel='sigmoid',degree= 15, C=1).fit(X_train, y_train)
svm.predict(X_test)
svm_accuracy = accuracy_score(svm.predict(X_test), y_test)
svm_roc = roc_auc_score(y_test, svm.predict(X_test))
print(f'SVM (sigmoid) accuracy: {svm_accuracy}; ROC: {svm_roc}')
print("----------------------------------------------------")

# Random Forest

nes = [100, 200, 400, 800, 1600]
depth = [5,10, 15, 25, 40]

results = {}

for n in nes: # run hyperparameter search
    results [n] = []

    for d in depth:   
    
        clf = RandomForestClassifier(max_depth=d, n_estimators=n, random_state=0).fit(X_train, y_train)
        rt_accuracy = accuracy_score(clf.predict(X_test), y_test)
        rt_roc = roc_auc_score(y_test, clf.predict(X_test))
        results[n].append([d, rt_accuracy, rt_roc])
        
        print(f'Random Forest ({n} estimators, {d} depth), accuracy: {round(rt_accuracy,5)}; ROC: {round(rt_roc,5)}')

print("----------------------------------------------------")
clf = RandomForestClassifier(max_depth=10, n_estimators=400, random_state=0)
clf.fit(X_train, y_train)
#100 200 400 800 1600

#depth 5,10, 15, 25, 40 

clf.predict(X_test)

rt_accuracy = accuracy_score(clf.predict(X_test), y_test)
rt_roc = roc_auc_score(y_test, clf.predict(X_test))

print(f'Random Forest(d={10},#est={400}), accuracy: {round(rt_accuracy,5)}, ROC: {round(rt_roc,5)}')
print("----------------------------------------------------")
# XGBoost

xgb_results = {}

for n in nes:
    
    xgb_results [n] = []
    
    for d in depth:
        
        clf = xgb.XGBClassifier(max_depth=d, n_estimators=n, random_state=0, use_label_encoder=False,verbosity=0).fit(X_train, y_train)
        rt_accuracy = accuracy_score(clf.predict(X_test), y_test)
        rt_roc = roc_auc_score(y_test, clf.predict(X_test))
        xgb_results[n].append([d, rt_accuracy, rt_roc])
        print(f'XGB ({n} estimators,  {d} depth), accuracy: {round(rt_accuracy,5)}, ROC: {round(rt_roc,5)}')    

print("----------------------------------------------------")
clf = xgb.XGBClassifier(max_depth=40, n_estimators=1600, random_state=0, use_label_encoder=False,verbosity=0).fit(X_train, y_train)
xgb_accuracy = accuracy_score(clf.predict(X_test), y_test)
xgb_roc = roc_auc_score(y_test, clf.predict(X_test))
predictions = clf.predict(X_test)

print(f"XGB (d={40}, #est={1600}), accuracy: {xgb_accuracy}; ROC: {xgb_roc}")
print("----------------------------------------------------")







