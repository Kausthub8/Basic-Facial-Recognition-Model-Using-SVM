import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
%matplotlib inline

import seaborn as sns; sns.set()


from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples = 50, centers = 2, random_state=0, cluster_std =0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')


from sklearn.datasets import fetch_lfw_people

from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca,svc)


from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.metrics import classification_report
