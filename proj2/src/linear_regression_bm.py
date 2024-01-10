import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pickle

training_set_norm = pd.read_csv('Dataset/training_set.csv', sep=';')
testing_set_norm = pd.read_csv('Dataset/testing_set.csv', sep=';')

X = pd.concat([training_set_norm.drop(['y'], axis=1), testing_set_norm.drop(['y'], axis=1)])
y = pd.concat([training_set_norm['y'], testing_set_norm['y']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

y_train_max = np.max(np.log10(10*y_train))
y_train = np.log10(10*y_train) / y_train_max
y_test = np.log10(10*y_test) / y_train_max

# Use the powertransformer to transform features
pt = PowerTransformer()
X = pt.fit_transform(X)
X_train = pt.fit_transform(X_train)
X_test = pt.transform(X_test)

### Train and save model with degree 1
poly_reg1 = PolynomialFeatures(degree=1)

# Transforming the training data
X_poly1 = poly_reg1.fit_transform(X_train)

# Training the model
pol_reg1 = LinearRegression()
pol_reg1.fit(X_poly1, y_train)

# Transforming the test data (not fitting)
X_test_poly1 = poly_reg1.transform(X_test)

# Predicting the Test set results
y_pred1 = pol_reg1.predict(X_test_poly1)

# Evaluating the model
# The mean squared error
print("Mean squared error: %.8f" % mean_squared_error(y_test, y_pred1))
# The mean absolute error
print("Mean absolute error: %.8f" % mean_absolute_error(y_test, y_pred1))
# Explained variance score: 1 is perfect prediction
print('Variance score (Train): %.8f' % r2_score(y_train, pol_reg1.predict(X_poly1)))
print('Variance score (Test): %.8f' % r2_score(y_test, y_pred1))

# Save the model on disk
filename = 'trained_models/pol_reg1.sav'
pickle.dump(pol_reg1, open(filename, 'wb'))
filename_pol_features = 'trained_models/pol_reg1_features.sav'
pickle.dump(poly_reg1, open(filename_pol_features, 'wb'))

### Train and save model with degree 2
poly_reg2 = PolynomialFeatures(degree=2)

# Transforming the training data
X_poly2 = poly_reg2.fit_transform(X_train)

# Training the model
pol_reg2 = LinearRegression()
pol_reg2.fit(X_poly2, y_train)

# Transforming the test data (not fitting)
X_test_poly2 = poly_reg2.transform(X_test)

# Predicting the Test set results
y_pred2 = pol_reg2.predict(X_test_poly2)

# Evaluating the model
# The mean squared error
print("Mean squared error: %.8f" % mean_squared_error(y_test, y_pred2))
# The mean absolute error
print("Mean absolute error: %.8f" % mean_absolute_error(y_test, y_pred2))
# Explained variance score: 1 is perfect prediction
print('Variance score (Train): %.8f' % r2_score(y_train, pol_reg2.predict(X_poly2)))
print('Variance score (Test): %.8f' % r2_score(y_test, y_pred2))

# Save the model on disk
filename = 'trained_models/pol_reg2.sav'
pickle.dump(pol_reg2, open(filename, 'wb'))
filename_pol_features = 'trained_models/pol_reg2_features.sav'
pickle.dump(poly_reg2, open(filename_pol_features, 'wb'))

### Train and save model with degree 3
poly_reg3 = PolynomialFeatures(degree=3)

# Transforming the training data
X_poly3 = poly_reg3.fit_transform(X_train)

# Training the model
pol_reg3 = LinearRegression()
pol_reg3.fit(X_poly3, y_train)

# Transforming the test data (not fitting)
X_test_poly3 = poly_reg3.transform(X_test)

# Predicting the Test set results
y_pred3 = pol_reg3.predict(X_test_poly3)

# Evaluating the model
# The mean squared error
print("Mean squared error: %.8f" % mean_squared_error(y_test, y_pred3))
# The mean absolute error
print("Mean absolute error: %.8f" % mean_absolute_error(y_test, y_pred3))
# Explained variance score: 1 is perfect prediction
print('Variance score (Train): %.8f' % r2_score(y_train, pol_reg3.predict(X_poly3)))
print('Variance score (Test): %.8f' % r2_score(y_test, y_pred3))

# Save the model on disk
filename = 'trained_models/pol_reg3.sav'
pickle.dump(pol_reg3, open(filename, 'wb'))
filename_pol_features = 'trained_models/pol_reg3_features.sav'
pickle.dump(poly_reg3, open(filename_pol_features, 'wb'))

### Train and save model with degree 4
poly_reg4 = PolynomialFeatures(degree=4)

# Transforming the training data
X_poly4 = poly_reg4.fit_transform(X_train)

# Training the model
pol_reg4 = LinearRegression()
pol_reg4.fit(X_poly4, y_train)

# Transforming the test data (not fitting)
X_test_poly4 = poly_reg4.transform(X_test)

# Predicting the Test set results
y_pred4 = pol_reg4.predict(X_test_poly4)

# Evaluating the model
# The mean squared error
print("Mean squared error: %.8f" % mean_squared_error(y_test, y_pred4))
# The mean absolute error
print("Mean absolute error: %.8f" % mean_absolute_error(y_test, y_pred4))
# Explained variance score: 1 is perfect prediction
print('Variance score (Train): %.8f' % r2_score(y_train, pol_reg4.predict(X_poly4)))
print('Variance score (Test): %.8f' % r2_score(y_test, y_pred4))

# Save the model on disk
filename = 'trained_models/pol_reg4.sav'
pickle.dump(pol_reg4, open(filename, 'wb'))
filename_pol_features = 'trained_models/pol_reg4_features.sav'
pickle.dump(poly_reg4, open(filename_pol_features, 'wb'))

### Train and save model with degree 5
poly_reg5 = PolynomialFeatures(degree=5)

# Transforming the training data
X_poly5 = poly_reg5.fit_transform(X_train)

# Training the model
pol_reg5 = LinearRegression()
pol_reg5.fit(X_poly5, y_train)

# Transforming the test data (not fitting)
X_test_poly5 = poly_reg5.transform(X_test)

# Predicting the Test set results
y_pred5 = pol_reg5.predict(X_test_poly5)

# Evaluating the model
# The mean squared error
print("Mean squared error: %.8f" % mean_squared_error(y_test, y_pred5))
# The mean absolute error
print("Mean absolute error: %.8f" % mean_absolute_error(y_test, y_pred5))
# Explained variance score: 1 is perfect prediction
print('Variance score (Train): %.8f' % r2_score(y_train, pol_reg5.predict(X_poly5)))
print('Variance score (Test): %.8f' % r2_score(y_test, y_pred5))

# Save the model on disk
filename = 'trained_modelspol_reg5.sav'
pickle.dump(pol_reg5, open(filename, 'wb'))
filename_pol_features = 'trained_modelspol_reg5_features.sav'
pickle.dump(poly_reg5, open(filename_pol_features, 'wb'))

### Train and save model with degree 6
poly_reg6 = PolynomialFeatures(degree=6)

# Transforming the training data
X_poly6 = poly_reg6.fit_transform(X_train)

# Training the model
pol_reg6 = LinearRegression()
pol_reg6.fit(X_poly6, y_train)

# Transforming the test data (not fitting)
X_test_poly6 = poly_reg6.transform(X_test)

# Predicting the Test set results
y_pred6 = pol_reg6.predict(X_test_poly6)

# Evaluating the model
# The mean squared error
print("Mean squared error: %.8f" % mean_squared_error(y_test, y_pred6))
# The mean absolute error
print("Mean absolute error: %.8f" % mean_absolute_error(y_test, y_pred6))
# Explained variance score: 1 is perfect prediction
print('Variance score (Train): %.8f' % r2_score(y_train, pol_reg6.predict(X_poly6)))
print('Variance score (Test): %.8f' % r2_score(y_test, y_pred6))

# Save the model on disk
filename = 'trained_models/pol_reg6.sav'
pickle.dump(pol_reg6, open(filename, 'wb'))
filename_pol_features = 'trained_models/pol_reg6_features.sav'
pickle.dump(poly_reg6, open(filename_pol_features, 'wb'))

### Train and save model with degree 7
poly_reg7 = PolynomialFeatures(degree=7)

# Transforming the training data
X_poly7 = poly_reg7.fit_transform(X_train)

# Training the model
pol_reg7 = LinearRegression()
pol_reg7.fit(X_poly7, y_train)

# Transforming the test data (not fitting)
X_test_poly7 = poly_reg7.transform(X_test)

# Predicting the Test set results
y_pred7 = pol_reg7.predict(X_test_poly7)

# Evaluating the model
# The mean squared error
print("Mean squared error: %.8f" % mean_squared_error(y_test, y_pred7))
# The mean absolute error
print("Mean absolute error: %.8f" % mean_absolute_error(y_test, y_pred7))
# Explained variance score: 1 is perfect prediction
print('Variance score (Train): %.8f' % r2_score(y_train, pol_reg7.predict(X_poly7)))
print('Variance score (Test): %.8f' % r2_score(y_test, y_pred7))

# Save the model on disk
filename = 'trained_models/pol_reg7.sav'
pickle.dump(pol_reg7, open(filename, 'wb'))
filename_pol_features = 'trained_models/pol_reg7_features.sav'
pickle.dump(poly_reg7, open(filename_pol_features, 'wb'))