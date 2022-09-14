
import pandas as pd
import csv as cv
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump


def functrainDecTree():
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    if paramDict['regression'] == 'yes':
        tree_model = DecisionTreeRegressor()
    else:
        tree_model = DecisionTreeClassifier()
    while score < 0.80:
        df = pd.read_csv('OracleData.csv')
        data = df.values
        X = data[:, :-1]
        y = data[:, -1]
        depth_list = [i for i in range(2, 2001)]
        random_state_list = [i for i in range(2, 501)]
        param_space = {"max_depth": [i for i in depth_list],
                       "criterion": ['gini', 'entropy'],
                       "max_features": ['auto', 'sqrt', 'log2', None],
                       "min_samples_leaf": [1, 2, 3, 4, 5],
                       "min_samples_split": [2, 3, 4, 5],
                       "random_state": [i for i in random_state_list]
                       }

        tree_rand_search = RandomizedSearchCV(tree_model, param_space, n_iter=50,
                                              scoring="accuracy", verbose=True, cv=5,
                                              n_jobs=-1, random_state=42)
        tree_rand_search.fit(X, y)
        score = tree_rand_search.best_score_
        if score < 0.80:
            genDataObj = generateData(feNameArr, feTypeArr, minValArr, maxValArr)
            genDataObj.funcGenerateTestData()
        print(tree_rand_search.best_score_)

    model = tree_rand_search.best_estimator_
    model = model.fit(X, y)
    dump(model, 'Model/dectree_approx.joblib')
    return model


