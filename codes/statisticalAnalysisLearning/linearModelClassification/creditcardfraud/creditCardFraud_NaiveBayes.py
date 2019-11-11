

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
plt.close('all')
mpl.rcParams.update({'font.size': 16})
#importing the data set
df=pd.read_csv("creditcard.csv")

import seaborn as sns
sns.set()

plt.close('all')

plt.figure(1, figsize=(15,15))

def sephist(col):
    yes = df[df['Class'] == 0][col]
    no = df[df['Class'] == 1][col]
    return yes, no

namesToPlot = [col for col in df.columns if col not in ['Class']]
densities = []
for num, alpha in enumerate(namesToPlot):
    plt.subplot(6, 5, num + 1)
    density, _, _ = plt.hist((df[alpha][df.Class ==0], df[alpha][df.Class ==1]), bins=25, alpha=0.5, label=['0', '1'], color=['r', 'b'], density=True)
    #plt.legend(loc='upper right')
    densities.append(density)
    plt.title(alpha)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

densitySimilarity = [np.dot(d1, d2)/np.linalg.norm(d1)/np.linalg.norm(d2) for (d1, d2) in densities]

sortSimilarity = pd.Series(densitySimilarity, namesToPlot).sort_values(ascending=False)


plt.figure(2, figsize=(6,6))
sortSimilarity.plot(kind='bar')
plt.ylabel('density similarity')



thresh = np.linspace(0.8, 1.0, 20)

accuracy = []


for th in thresh:
    dropList = [col for col, p in zip(namesToPlot, densitySimilarity) if p > th]
    dropSimilarity = [p for col, p in zip(namesToPlot, densitySimilarity) if p > th]
    #g = sns.FacetGrid(df, hue='Class')
    X = df[namesToPlot].drop(dropList, axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify =y)
    
    
    clf_nb = GaussianNB()
    
    clf_nb.fit(X_train, y_train)
    
    y_pred = clf_nb.predict(X_test)
    y_pred_prob = clf_nb.predict_proba(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('balanced accuracy score', balanced_accuracy_score(y_test, y_pred))
    print(roc_auc_score(1 - y_test, y_pred_prob[:,0]))
    print(roc_auc_score(y_test, y_pred_prob[:,1]))
    accuracy.append(balanced_accuracy_score(y_test, y_pred))
    
    
plt.figure(3, figsize=(6,6))

plt.plot(thresh, accuracy)
plt.xlabel('thresh')
plt.ylabel('accuracy score')