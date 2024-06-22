import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from urllib.request import urlretrieve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# load dataset
dataset = pd.read_csv('ETH_Hack/models/iris.csv', sep=',')
dataset = dataset.rename(columns=lambda x: x.strip().lower())
dataset.head()

# cleaning missing values

y = dataset['species']
# sepal_length,sepal_width,petal_length,petal_width,species
dataset = dataset[['sepal_length','sepal_width','petal_length','petal_width']]


# scaling features 
sc = MinMaxScaler(feature_range=(0,1))
X_scaled = sc.fit_transform(dataset)

# model fit
log_model = LogisticRegression(C=1)
log_model.fit(X_scaled, y)

# saving model as a pickle
import pickle
pickle.dump(log_model,open("titanic_survival_ml_model.sav", "wb"))
pickle.dump(sc, open("scaler.sav", "wb"))