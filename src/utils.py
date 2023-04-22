import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score


def save_object(file_path, obj):
    try:
        # Check if the file already exists
        if os.path.exists(file_path):
            # If it does, remove the file
            os.remove(file_path)

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict_proba(X_train)

            y_test_pred = model.predict_proba(X_test)

            train_model_score = roc_auc_score(y_train, y_train_pred[:,1])

            test_model_score = roc_auc_score(y_test, y_test_pred[:,1])

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)