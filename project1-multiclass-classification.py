# Python Project Template
# 1. Prepare Problem
#   a) Load libraries
        # Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#   b) Load dataset
        # Load dataset
filename = 'datasets/iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)
#print(dataset)

# 2. Summarize Data
#   a) Descriptive statistics
        # shape
""" print(dataset.shape)
        # head
print(dataset.head(20))
        # descriptions
print(dataset.describe())
        # class distribution
print(dataset.groupby('class').size()) """

#   b) Data visualizations
        # box and whisker plots
""" dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
        # histograms
dataset.hist()
pyplot.show()
        # scatter plot matrix
scatter_matrix(dataset)
pyplot.show() """

# 3. Prepare Data
#   a) Data Cleaning
#   b) Feature Selection
#   c) Data Transforms

# 4. Evaluate Algorithms
#   a) Split-out validation dataset
    # Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)

#   b) Test options and evaluation metric
#   c) Spot Check Algorithms
# Spot-Check Algorithms
""" models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg) """

#   d) Compare Algorithms
# Compare Algorithms
""" fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show() """


# 5. Improve Accuracy
#   a) Algorithm Tuning
#   b) Ensembles

# 6. Finalize Model
#   a) Predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


#   b) Create standalone model on entire training dataset
#   c) Save model for later use
