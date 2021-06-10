import sklearn
import numpy as np
from sklearn.svm import SVC
import idx2numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.decomposition import PCA
# stat functions
from scipy.stats import uniform, norm
# to measure time
from time import time




# loading MNIST Dataset (source: http://yann.lecun.com/exdb/mnist/)
x_train_file = 'train-images.idx3-ubyte'
x_train = idx2numpy.convert_from_file(x_train_file)
# reshaping it in order to process it
x_train = x_train.reshape(60000, -1)



y_train_file = 'train-labels.idx1-ubyte'
y_train = idx2numpy.convert_from_file(y_train_file)
y_train = pd.DataFrame(y_train)
# making the binary class problem
y_train = y_train.replace([0, 2, 4, 6, 8], [0, 0, 0, 0, 0])
y_train = y_train.replace([1, 3, 5, 7, 9], [1, 1, 1, 1, 1])
y_train = np.ravel(y_train)



x_test_file = 't10k-images.idx3-ubyte'
x_test = idx2numpy.convert_from_file(x_test_file)
x_test = x_test.reshape(10000, -1)


y_test_file = 't10k-labels.idx1-ubyte'
y_test = idx2numpy.convert_from_file(y_test_file)
y_testt = idx2numpy.convert_from_file(y_test_file)
y_test = pd.DataFrame(y_test)
# making the binary class problem
y_test = y_test.replace([0, 2, 4, 6, 8], [0, 0, 0, 0, 0])
y_test = y_test.replace([1, 3, 5, 7, 9], [1, 1, 1, 1, 1])
y_test = np.ravel(y_test)

# ==========================================================================
# preprocessing data

# visualise a number:
number = x_train[1, :]
number = number.reshape(28, 28)
plt.imshow(number, cmap='gray')
plt.show()


# applying standardization:
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# ==========================================================================
# performing pca for dimensionality reduction

# making an object from PCA class

pca_old = PCA()
x_train1 = pca_old.fit(x_train)

total = sum(pca_old.explained_variance_)

var_explained = [(i/total)*100 for i in sorted(pca_old.explained_variance_, reverse=True)]

cumulative_var_explained = np.cumsum(var_explained)

# plotting the explained variances:
plt.figure(figsize=(10, 5))
plt.step(range(1, 785), cumulative_var_explained, where='mid', label='cumulative explained variance')
plt.title('Cumulative Explained Variance while changing N of components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=90, color='red', linestyle='--', label='95% Explained Variance')
plt.axhline(y=85, color='c', label='85% Explained Variance')
plt.axhline(y=80, color='blue', linestyle='--', label='80% Explained Variance')
plt.legend(loc='best')
plt.savefig('CumulativeExplained Variance while changing N of components.png')


# choosing number of components that cover 90% of the variance:
comp = np.argmax(cumulative_var_explained > 90.0)

#setting time for start PCA
t = time()

# transforming data:
pca = PCA(n_components=comp)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

print('PCA done in: %0.3fs' % (time()-t))


# how much of the total variance is explained:
print('The percentage of variance that is explained is :', pca.explained_variance_ratio_.sum())


# ==========================================================================
# constructing linear problem
linear_classifier = SVC(random_state=42, kernel='linear')

# setting time for linear problem:
tl = time()

linear_classifier.fit(x_train_pca, y_train)

print('Linear problem constructed in: %0.3fs' % (time()-tl))

# predicting
linear_predicted = linear_classifier.predict(x_test_pca)

# model evaluation
print('Accuracy in linear problem is: ', sklearn.metrics.accuracy_score(y_test, linear_predicted))

# print class wise predictions
print(sklearn.metrics.classification_report(y_test, linear_predicted))
# ==========================================================================

# constructing non linear problem
classifier = SVC(random_state=42, kernel='rbf')
classifier.fit(x_train, y_train)

# predicting
predicted = classifier.predict(pca.transform(x_test))

# model evaluation
print('Accuracy without grid search is (default values are C=1, gamma= 1/784*Var(X)):', sklearn.metrics.accuracy_score(y_test, predicted))

# print class wise predictions
print(sklearn.metrics.classification_report(y_test, predicted))

# ==========================================================================

# exhausting grid search and cross validation for hyperparameter tuning
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

parameters = {'C': [1, 10],
              'gamma': [1e-3, 1e-4]}

svc_grid_search = SVC(kernel='rbf')

# setting starting time of gridsearch
t0 = time()

clf = GridSearchCV(estimator=svc_grid_search, param_grid=parameters, scoring='accuracy', cv=kfolds)

clf.fit(x_train_pca, y_train)

print('Training on exhausted grid search done in %0.3fs' % (time()-t0))

# data frame with cv results
cv_results = pd.DataFrame(clf.cv_results_)
print(cv_results)

# choosing the best hyper-parameters
best_par = clf.best_params_
C = best_par['C']
gamma = best_par['gamma']
print('The best hyperparameters for the exhausted grid search are:', clf.best_params_)

# accuracy score for grid search
print('Accuracy score for exahusted grid search is: ', sklearn.metrics.accuracy_score(y_test, clf.predict(x_test_pca)))

# class wise scores:
print(sklearn.metrics.classification_report(y_test, clf.predict(x_test_pca)))

# building the model with the optimal hyperparameters:

best_clf = SVC(C=C, gamma=gamma, kernel='rbf')

best_clf.fit(x_train, y_train)
y_pred = best_clf.predict(pca.transform(x_test))

#evaluating model

print('Accuracy of the best model is: ', sklearn.metrics.accuracy_score(y_test, y_pred))

# ==========================================================================

# randomized grid search and cross validation fot hyperparameter tuning
rparameters = dict(C=norm(loc=1, scale=0.5),
              gamma=uniform(loc=0.001, scale=0.0003))

svc_randomgrid_search = SVC(kernel='rbf', random_state=42)

# setting starting time of random grid search
t1 = time()

rclf = RandomizedSearchCV(svc_randomgrid_search, param_distributions=rparameters, scoring='accuracy', cv=5, n_iter=5)

random_class = rclf.fit(x_train_pca, y_train)

print('Training on randomized grid search done in %0.3fs' % (time()-t1))

# accuracy score for randomizer grid search
print('Accuracy score for randomized grid search is: ', sklearn.metrics.accuracy_score(y_test, rclf.predict(x_test_pca)))
# best parameters after the randomized grid search
print('The best values for C and gamma,after randomized search, are: ', random_class.best_params_)

# ==========================================================================






