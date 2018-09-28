# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 12:16:28 2018

@author: yuguangyang
"""

import numpy as np
from sklearn.preprocessing import Imputer
X = [[np.nan, 2], [6, np.nan], [7, 6], [6, 6]]

# mean imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X1 = imp.fit_transform(X)


# median imputer
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
X2 = imp.fit_transform(X)

# most_frequent imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
X3 = imp.fit_transform(X)


