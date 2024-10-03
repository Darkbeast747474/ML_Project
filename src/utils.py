import os, sys, dill
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException
from src.logger import logging


def save_obj(obj, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        logging.info(e)
        raise CustomException(e, sys)
    
def load_obj(path):
    try:
        with open(path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        logging.info(e)
        raise CustomException(e, sys)


class ModelSelector:
    def __init__(self, models: dict):
        self.models = models
        self.best_model_name = None
        self.best_score = -float("inf")
        self.best_model = None
        self.model_scores = {}
        self.param_grid = {
                "RandomForestClassifier": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False],
                },
                "LogisticRegression": {
                    "penalty": ["l1", "l2", "elasticnet", None],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "solver": ["lbfgs", "liblinear", "saga"],
                    "max_iter": [100, 200, 300],
                },
                "XGBClassifier": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "gamma": [0, 0.1, 0.3],
                },
                "GradientBoostingClassifier": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "AdaBoostClassifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                    "algorithm": ["SAMME", "SAMME.R"],
                },
                "DecisionTreeClassifier": {
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"],
                },
                "KNeighborsClassifier": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"],
                },
            }


    def fit_models(self, X_train, y_train, X_test, y_test):
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            self.model_scores[model_name] = score
            print(f"{model_name}: {score}")

            if score > self.best_score:
                self.best_score = score
                self.best_model_name = model_name
                self.best_model = model

        return self.model_scores

    def get_best_model(self):
        return self.best_model

    def get_best_model_name(self):
        return self.best_model_name

    def get_best_score(self):
        return self.best_score 
    
    def Tune_best_model(self,best_model_name,X_train,y_train):
        param = self.param_grid[best_model_name]
        rscv = RandomizedSearchCV(self.best_model, param, cv=2, n_jobs=-1,).fit(X_train, y_train)
        return self.models[best_model_name].set_params(**rscv.best_params_).fit(X_train, y_train),rscv.best_score_