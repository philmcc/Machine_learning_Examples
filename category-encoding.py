import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('datasets/houseprices/train.csv')
#print(df.head())
#print(df.dtypes)
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
#print("Categorical variables:")
#print(object_cols)

features = df[['Type','Method','Regionname']]
print(features.head())

#print(features.Method.value_counts())
#dfmethod = pd.get_dummies(features['Method'])
#dftype = pd.get_dummies(features['Type'])

dfregion = pd.get_dummies(features['Regionname'])
df3 = []
df3.append(dfmethod)
df3.append(dfregion)
df3.append(dftype)


#print(dftype)

