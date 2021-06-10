import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.svm import SVC
# for statistics
from scipy.stats import uniform, norm


# source of data: https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018 / (i used the xls for 2018)
financial_data = pd.read_csv('2018_Financial_Data.csv')

# for computation reasons, i'm randomly sampling 3000 instatnces from the data.
first_data = financial_data.sample(n=3000, random_state=42)
print(first_data.head())
# also for computation reasons, i will select 15 columns based on the Pearson's correlation with the variable Class:
correlated = first_data.corr()
correlated = pd.DataFrame(correlated)
# these are the variables with the 5 bigger correlation and also the variable class:
print(correlated.nlargest(15, ['Class']))
# shape them as a new data frame:

data = first_data[['10Y Net Income Growth (per Share)', '2019 PRICE VAR [%]', '10Y Operating CF Growth (per Share)', 'Dividend per Share', 'cashConversionCycle', '10Y Revenue Growth (per Share)', '10Y Shareholders Equity Growth (per Share)', '5Y Net Income Growth (per Share)', 'Gross Margin', '10Y Dividend per Share Growth (per Share)', 'EBIT', 'Total non-current liabilities', '5Y Dividend per Share Growth (per Share)', 'Total non-current assets', 'Class']]

# data preprocessing:
# firstly we will see how many of the  data has missing values:
print(data.isnull().sum())

# then calculate the variables means:
m1 = data['10Y Net Income Growth (per Share)'].mean()
m2 = data['2019 PRICE VAR [%]'].mean()
m3 = data['10Y Operating CF Growth (per Share)'].mean()
m4 = data['Dividend per Share'].mean()
m5 = data['10Y Revenue Growth (per Share)'].mean()
m6 = data['cashConversionCycle'].mean()
m7 = data['10Y Shareholders Equity Growth (per Share)'].mean()
m8 = data['5Y Net Income Growth (per Share)'].mean()
m9 = data['Gross Margin'].mean()
m10 = data['10Y Dividend per Share Growth (per Share)'].mean()
m11 = data['EBIT'].mean()
m12 = data['Total non-current liabilities'].mean()
m13 = data['5Y Dividend per Share Growth (per Share)'].mean()
m14 = data['Total non-current assets'].mean()


# then fill na's with those means:
values = {'10Y Net Income Growth (per Share)': m1, '2019 PRICE VAR [%]': m2, '10Y Operating CF Growth (per Share)': m3, 'Dividend per Share': m4, '10Y Revenue Growth (per Share)': m5, 'cashConversionCycle': m6, '10Y Shareholders Equity Growth (per Share)': m7, '5Y Net Income Growth (per Share)': m8, 'Gross Margin': m9, '10Y Dividend per Share Growth (per Share)': m10, 'EBIT': m11, 'Total non-current liabilities': m12, '5Y Dividend per Share Growth (per Share)': m13, 'Total non-current assets': m14}
data = data.fillna(value=values)

# check that values replaced:
print(data.isnull().sum())

# ===================================================================================

# in this section we could apply pca, but the variables are already small in number so we will continue on splitting the dataset.
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# splitting data on train(0,6) and test(0,4):
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(y_test)

# Data standardiazation:
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ===================================================================================

# from this point we will apply svm's in order to classificate the data:

# constructing linear problem
linear_classifier = SVC(random_state=42, kernel='linear')

# setting time for linear problem:
tl = time()

linear_classifier.fit(x_train, y_train)

print('Linear problem constructed in: %0.3fs' % (time()-tl))

# predicting
linear_predicted = linear_classifier.predict(x_test)

# model evaluation
print('Accuracy in linear problem is: ', sklearn.metrics.accuracy_score(y_test, linear_predicted))

# print class wise predictions
print(sklearn.metrics.classification_report(y_test, linear_predicted))

# ===================================================================================
# constructing non linear problem with:
# exhausting grid search and cross validation for hyperparameter tuning

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

parameters = {'C': [1, 5, 10, 15],
              'gamma': [1e-5, 1e-6, 1e-4, 1e-3, 1e-2]}

svc_grid_search = SVC(kernel='rbf', random_state=42)

# setting starting time of gridsearch
t0 = time()

clf = GridSearchCV(estimator=svc_grid_search, param_grid=parameters, scoring='accuracy', cv=kfolds)

clf.fit(x_train, y_train)

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
print('Accuracy score for exahusted grid search is: ', sklearn.metrics.accuracy_score(y_test, clf.predict(x_test)))

# class wise scores:
print(sklearn.metrics.classification_report(y_test, clf.predict(x_test)))

# =====================================================================================================================

# constructing non linear problem:
# randomized grid search and cross validation fot hyperparameter tuning:

rparameters = dict(C=norm(loc=10, scale=5),
              gamma=uniform(loc=0.001, scale=0.000003))

svc_randomgrid_search = SVC(kernel='rbf', random_state=42)

# setting starting time of random grid search
t1 = time()

rclf = RandomizedSearchCV(svc_randomgrid_search, param_distributions=rparameters, scoring='accuracy', cv=kfolds, n_iter=10, random_state=42)

random_class = rclf.fit(x_train, y_train)

print('Training on randomized grid search done in %0.3fs' % (time()-t1))

# accuracy score for randomizer grid search
print('Accuracy score for randomized grid search is: ', sklearn.metrics.accuracy_score(y_test, rclf.predict(x_test)))
# best parameters after the randomized grid search
print('The best values for C and gamma,after randomized search, are: ', random_class.best_params_)

print(sklearn.metrics.classification_report(y_test, rclf.predict(x_test)))

# =====================================================================================================================

# cross validation for the linear kernel: (I could not run it, but this would be in the place of the first linear kernel model)
for train_index, test_index in kfolds.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    tl1 = time()

    linear_classifier.fit(x_train, y_train)

    print('Linear problem with 5-fold cross validation constructed in: %0.3fs' % (time() - tl1))

    # predicting
    linear_predicted = linear_classifier.predict(x_test)

    # model evaluation
    print('Accuracy in linear problem is: ', sklearn.metrics.accuracy_score(y_test, linear_predicted))

# =====================================================================================================================
