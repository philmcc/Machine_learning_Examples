# import 
import numpy as np
import pandas as pd
from sklearn import preprocessing

# load dataset
X = pd.read_csv('datasets/houseprices/train.csv')
#print(X.head(3))
print(X.dtypes)
print(X.shape)
# limit to categorical data using df.select_dtypes()
X = X.select_dtypes(include=[object])
print(X.dtypes)
#print(X.head(3))
print(X.shape)
print(X.columns)

""" X.pop('Address')
X.pop('SellerG')
X.pop('Date') """




#print(X.columns)
# TODO: create a LabelEncoder object and fit it to each feature in X


# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()
# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
X_2 = X.apply(le.fit_transform)#
#print(X_2.head())

# TODO: create a OneHotEncoder object, and fit it to all of X

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_2)

# 3. Transform
onehotlabels = enc.transform(X_2).toarray()
print(onehotlabels.shape)

# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data

#print(onehotlabels)

