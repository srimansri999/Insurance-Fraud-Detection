from src.exception.exception import CustomException
import sys
import yaml
import os
from typing import Any
import numpy as np
from src.logging.logger import logging
import pickle
from sklearn.model_selection import  ParameterGrid, RandomizedSearchCV
from sklearn.metrics import recall_score, roc_auc_score


def read_yaml_file (file_path: str)-> dict:
    try:
        with open(file_path,'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)

def write_yaml_file(
    file_path: str,
    content: Any,
    replace: bool = False
) -> None:

    try:

        if replace and os.path.exists(file_path):
            os.remove(file_path)

        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as file:
            yaml.safe_dump(content, file)

    except Exception as e:
        raise CustomException(e, sys)

def save_numpy_array_data(file_path: str, array: np.array):
    '''
    Save numpy array data to file
    file_path: str - location of file to save
    array: np.array  - data to save
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException (e,sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils Class")

        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Exited the save_object method of MainUtils Class")

    except Exception as e:
        raise CustomException(e, sys) from e

def load_numpy_array (file_path:str)-> np.array:
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params,
    scoring="recall"):
    """
    Perform GridSearchCV for multiple models and evaluate them.

    Returns:
        report (dict): {
            model_name: {
                "recall": float,
                "auc": float
            }
        }

        best_models (dict): {
            model_name: trained_best_estimator
        }
    """

    try:

        report = {}
        best_models = {}

        logging.info("Starting model evaluation.")

        for model_name, model in models.items():

            logging.info(f"Training {model_name}")

            param_grid = params[model_name]
            n_iter = min(20, len(ParameterGrid(param_grid)))

            #gs = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring=scoring,n_jobs=-1,verbose=1)
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )

            #gs.fit(X_train, y_train)
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_

            #best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)

            # Get probability/decision scores for ROC-AUC
            if hasattr(best_model, "predict_proba"):
                y_test_prob = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_test_prob = best_model.decision_function(X_test)
            else:
                raise ValueError(
                    f"{model_name} does not support probability estimation."
                )

            recall = recall_score(y_test, y_test_pred)
            auc = roc_auc_score(y_test, y_test_prob)

            report[model_name] = {
                "recall": recall,
                "auc": auc
            }

            best_models[model_name] = best_model

            #logging.info(f"{model_name} Best Parameters : {gs.best_params_}")
            #logging.info(f"{model_name} Cross Validation Recall : {gs.best_score_:.4f}")
            logging.info(f"{model_name} Best Parameters : {best_params}")
            logging.info(f"{model_name} Cross Validation Recall : {random_search.best_score_:.4f}")
            logging.info(f"{model_name} Test Recall : {recall:.4f}")
            logging.info(f"{model_name} Test ROC-AUC : {auc:.4f}")

        logging.info("Model evaluation completed successfully.")

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException (e,sys) from e


