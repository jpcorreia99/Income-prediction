import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


x_train = pd.read_pickle("preprocessed_data/x_train")
y_train = pd.read_pickle("preprocessed_data/y_train").values.ravel() # transforms the collumns in a np array with shape (nsamples,)
x_test = pd.read_pickle("preprocessed_data/x_test")
y_test = pd.read_pickle("preprocessed_data/y_test").values.ravel()

forest_reg = RandomForestClassifier()

'''param_grid = [
    {'n_estimators':[10,50,100,200],'max_features':[3,6,9,12]},
]'''

'''param_grid = [
    {'n_estimators':[200,400,600],'max_features':[3]},
]'''

'''param_grid = [
    {'n_estimators':[350,400,450],'max_features':[3]},
]'''

param_grid = [
    {'n_estimators':[450],'max_features':[3]},
]


grid_search = GridSearchCV(forest_reg, param_grid,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)

grid_search.fit(x_train, y_train)
print(grid_search.best_params_)


best_estimator = grid_search.best_estimator_
best_estimator.fit(x_train,y_train)
print("score: ", best_estimator.score(x_test,y_test))

#first result {'max_features': 3, 'n_estimators': 200}
#second result {'max_features': 3, 'n_estimators': 400} score:  0.8448207171314741
#third result {'max_features': 3, 'n_estimators': 450} score:  0.8462815405046481

#ENSEMBLE METHODS
feature_importances = grid_search.best_estimator_.feature_importances_
features = ["age","workclass","education_num","marital_status","occupation","relationship","race","sex",
          "capital_gain","capital_loss","hours_per_week","native_country"]
print(sorted(zip(feature_importances, features), reverse=True))

#we see that race and sex don't matter that much, let's drop them and see how it improves
x_train = x_train.drop(["race","sex"], axis = 1)
x_test = x_test.drop(["race","sex"], axis = 1)
best_estimator.fit(x_train,y_train)
print("score: ", best_estimator.score(x_test,y_test))

#the score is improved minimally
import pickle
#pickle.dump(best_estimator, open("best_estimator.pickle", "wb"))

# Randomized Search example
'''
from sklearn.model_selection import RandomizedSearchCV
param_dist = [{'n_estimators': range(1,100), 'max_features': range(1,8)}]

grid_search = RandomizedSearchCV(forest_reg, param_dist, n_iter=3,
                                   scoring="neg_mean_squared_error",
                                   return_train_score=True)
'''