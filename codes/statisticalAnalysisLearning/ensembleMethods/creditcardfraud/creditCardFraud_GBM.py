import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
plt.close('all')
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

plt.figure(2)

plt.plot(densitySimilarity)

sortSimilarity = pd.Series(densitySimilarity, namesToPlot).sort_values(ascending=False)

plt.figure(3, figsize=(6,6))
sortSimilarity.plot(kind='bar')
plt.ylabel('density similarity')

X = df[namesToPlot]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify =y)


clf_gbm = GradientBoostingClassifier(random_state=1)
clf_gbm.fit(X_train,y_train)

y_pred_gbm = clf_gbm.predict(X_test)

feat_imp = pd.Series(clf_gbm.feature_importances_, namesToPlot).sort_values(ascending=False)
plt.figure(4, figsize=(6,6))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

print(classification_report(y_test, y_pred_gbm))
print(confusion_matrix(y_test, y_pred_gbm))

clf_dt = DecisionTreeClassifier(random_state=1)
clf_dt.fit(X_train, y_train)

y_pred_dt = clf_dt.predict(X_test)


print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))