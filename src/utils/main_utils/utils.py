from src.exception.exception import CustomException
import sys
import yaml
import os
from typing import Any
import numpy as np
from src.logging.logger import logging
import pickle


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


