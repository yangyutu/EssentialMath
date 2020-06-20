import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.cm as cm
mpl.rcParams['image.cmap'] = 'jet'
mpl.rcParams.update({'font.size': 16})
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

plt.figure(3, figsize=(6,10))
sortSimilarity.plot(kind='bar')
plt.ylabel('density similarity')

X = df[namesToPlot]
y = df['Class']

sd = StandardScaler()
X = sd.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify =y)


rus = RandomUnderSampler(random_state=1)
X_train, y_train = rus.fit_resample(X_train, y_train)

clf_lr_base = LogisticRegression(class_weight='balanced', solver='saga', max_iter=5000)
clf_lr_base.fit(X_train, y_train)
y_pred_lr_base = clf_lr_base.predict(X_test)


print(classification_report(y_test, y_pred_lr_base))
print(confusion_matrix(y_test, y_pred_lr_base))
print(balanced_accuracy_score(y_test, y_pred_lr_base))
parameter = {'C': np.logspace(-6, 2, 10)}

gs = GridSearchCV(LogisticRegression(solver='saga', max_iter=5000, penalty='l1', class_weight='balanced'), parameter, scoring='balanced_accuracy')

gs.fit(X_train, y_train)

y_pred_lr_L1 = gs.best_estimator_.predict(X_test)

print(classification_report(y_test, y_pred_lr_L1))
print(confusion_matrix(y_test, y_pred_lr_L1))
print(balanced_accuracy_score(y_test, y_pred_lr_L1))

C = np.logspace(-6, 2, 10)

clf_lr_L1Path = LogisticRegression(solver='saga', max_iter=5000, penalty='l1', class_weight='balanced')
coeff = []
scores = []
for c in C:
    clf_lr_L1Path.set_params(C=c)
    clf_lr_L1Path.fit(X_train, y_train)
    coeff.append(clf_lr_L1Path.coef_[0])
    y_pred = clf_lr_L1Path.predict(X_test)
    scores.append(balanced_accuracy_score(y_test, y_pred))
    
coeff = np.array(coeff)
color=cm.rainbow(np.linspace(0,1,coeff.shape[1]))
fig = plt.figure(6, figsize=(7, 10))
ax = plt.gca()
for coef, c in zip(coeff.T, color):
    ax.plot(np.log10(C), coef, c=c)
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('Logistic Regression Path')
plt.legend(namesToPlot, loc=6)

fig = plt.figure(7, figsize=(5, 5))
ax = plt.gca()
ax.plot(np.log10(C), scores)
plt.xlabel('log(C)')
plt.ylabel('balanced accuracy')

