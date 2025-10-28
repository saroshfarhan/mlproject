import os
import sys

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
import dill
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj: The Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object at {file_path}")
        raise CustomException(e, sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple regression models and return their R2 scores.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target.
        models (dict): A dictionary of model names and their corresponding instantiated model objects.

    Returns:
        dict: A dictionary with model names as keys and their R2 scores as values.
    """
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[model_name] = r2
            logging.info(f"{model_name} evaluated with R2 Score: {r2}")
        return model_report
    except Exception as e:
        logging.error("Error during model evaluation")
        raise CustomException(e, sys)    