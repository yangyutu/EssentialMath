# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../data')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from cleanNewsGroup import getCountVec
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
# fetch a subset of news
categories = ['rec.baseall', 'soc.religion.christian', 'comp.graphics', 'sci.space','talk.politics.guns']
# loading all files as training data. 
X, y, labelName, kvocab = getCountVec('../../data/20_newsgroups/', 2000, categories)


tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X_tf = tf_transformer.transform(X)

print(X_tf.shape)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)
print(X_tf.shape)

# latent semantic analysis

TD = Normalizer().transform(X_tfidf)
K = 8
svd = TruncatedSVD(n_components=K)

W = svd.fit_transform(TD)

components = svd.components_

for i in range(K):
    index = components[i].argsort()[-20:][::-1]
    feature = list(X.columns)
    print([feature[i] for i in index])
    
singular = svd.singular_values_
plt.figure(1, figsize=(8,8))
plt.plot(singular)
plt.xlabel('k')
plt.ylabel('singular value')
