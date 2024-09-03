import os
import pickle
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
import logging
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e, sys)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            print(f"Training model: {model_name}")
            para = param[model_name]
            print(f"Parameters: {para}")

            rs = RandomizedSearchCV(model, para, n_iter=3, cv=3, verbose=2, random_state=42, n_jobs=1)
            
            try:
                rs.fit(X_train, y_train)
            except Exception as e:
                print(f"Error fitting model {model_name}: {str(e)}")
                print(f"Model: {model}")
                print(f"Parameters: {para}")
                continue

            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            report[model_name] = test_accuracy

            logging.info(f"{model_name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)