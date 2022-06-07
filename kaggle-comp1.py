# Python Project Template

# 1. Prepare Problem
#   a) Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
#   b) Load dataset
url = 'datasets\houseprices/train.csv'
dataset = read_csv(url)
# 2. Summarize Data
#   a) Descriptive statistics
""" # shape
print(dataset.shape)
# types
print(dataset.dtypes)
# head
print(dataset.head(20))
# descriptions
set_option('precision', 1)
print(dataset.describe())
# correlation
set_option('precision', 2)
print(dataset.corr(method='pearson')) """

#   b) Data visualizations

# 3. Prepare Data
#   a) Data Cleaning
#   b) Feature Selection
#   c) Data Transforms

# 4. Evaluate Algorithms
#   a) Split-out validation dataset
array = dataset.values
X = array[:,0:37]
Y = array[:,37]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)
#   b) Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_log_error'
#   c) Spot Check Algorithms
#   d) Compare Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# 5. Improve Accuracy
#   a) Algorithm Tuning
#   b) Ensembles

# 6. Finalize Model
#   a) Predictions on validation dataset
#   b) Create standalone model on entire training dataset
#   c) Save model for later use
