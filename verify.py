from sklearn import datasets,linear_model

import numpy as np

from sklearn.model_selection import cross_val_predict, cross_val_score

diabetes = datasets.load_diabetes()

X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()

scores = cross_val_score(lasso, X, y, cv=3)
scores = np.array(scores)
mean_score = scores.mean()
print("Mean Score : " + str(mean_score))
y_pred = cross_val_predict(lasso, X, y, cv=3)
print(y_pred)