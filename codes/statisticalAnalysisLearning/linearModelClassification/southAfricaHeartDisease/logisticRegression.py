import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import transforms, pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 16})
plt.close('all')
# set common plots properties in advance
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
plt.rc('text', usetex=False)
GRAY4, GRAY7 = '#646369', '#929497'

# load training and test data set from a text file
df = pd.read_csv("South African Heart Disease.txt")
# encode famhist with dummy 0-1 variable
df['famhist'] = pd.get_dummies(df['famhist'])['Present']
target = 'chd'
features = ['sbp', 'tobacco', 'ldl', 'famhist', 'obesity', 'alcohol', 'age']
df[features + [target]].head()
df.describe()

sns.pairplot(
    df, vars=features, kind="scatter", hue=target,
    palette=['#22FF00', '#FF3300'], plot_kws=dict(s=30, linewidth=1)
)._legend.remove()


# convert data to X, y np.arrays
X, y = df[features].values, df[target].values

import statsmodels.api as sm
from scipy import stats

lr = sm.Logit(y, sm.add_constant(X)).fit(disp=False)

# PAGE 122. TABLE 4.2. Results from a logistic regression fit to the South
#           African heart disease data.
#           There are some surprises in this table of coefficients, which must
#           be interpreted with caution. Systolic blood pressure (sbp) is not
#           significant! Nor is obesity, and its sign is negative. This
#           confusion is a result of the correlation between the set of
#           predictors. On their own, both sbp and obesity are significant, and
#           with positive sign. However, in the presence of many other
#           correlated variables, they are no longer needed (and can even get a
#           negative sign).
result = zip(['(Intercept)'] + features, lr.params, lr.bse, lr.tvalues)
print('               Coefficient   Std. Error   Z Score')
print('-------------------------------------------------')
for term, coefficient, std_err, z_score in result:
    print(f'{term:>12}{coefficient:>14.3f}{std_err:>13.3f}{z_score:>10.3f}')
    
from sklearn.preprocessing import StandardScaler



lr = sm.Logit(y, sm.add_constant(StandardScaler().fit_transform(X)))
# for different alpha values fit a model and save coefficients
alpha = np.linspace(0, 82, 100)
coefs = np.vstack([lr.fit_regularized(disp=False, alpha=a).params[1:]
                   for a in alpha])
# calculate sum of coefficients for different alpha values
coefs_l1_norm = np.sum(np.abs(coefs), axis=1)

fig, ax1 = plt.subplots(figsize=(5, 5), dpi=150)
colors = ['#000101', '#FF9234', '#29D0D0', '#85C57A', '#57575A', '#AD2323',
          '#2A4BD7']
for i in range(7):
    ax1.plot(coefs_l1_norm, coefs[:, i], color=colors[i])
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
plt.setp(ax2, yticks=coefs[0], yticklabels=features)
for i in ax1.get_yticklabels() + ax1.get_xticklabels() + ax2.get_yticklabels():
    i.set_fontsize(7)
ax1.set_xlabel(r'$||\beta(\lambda)||_1$', color=GRAY4, fontsize=9)
_ = ax1.set_ylabel(r'$\beta_j(\lambda)$', color=GRAY4, fontsize=9)