import os
from src.logger import get_logger
import pickle
from src.exception import CustomException
import sys
import json
import numpy as np
logger = get_logger('Main_utils')
def save_object(object, file):

    try: 
        logger.info('Saving Object')

        with open(file, 'wb') as File: 
            pickle.dump(object, File)
    except Exception as e: 
        logger.error("Something Went Wrong while saving preprocessor")
        raise CustomException(e, sys)

logger = get_logger("ConfigLoader")
def load_config(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, "r") as f:
            config = json.load(f)
        
        logger.info(f"Config loaded successfully from {file_path}")
        return config
    
    except Exception as e:
        logger.error(f"Failed to load config from {file_path}")
        raise CustomException(e, sys)
def save_numpy_array_data(file_path: str, array: np.array):
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) 
    
def load_numpy_array_data(file_path: str) -> np.array:


    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e