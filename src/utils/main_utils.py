import os
from src.logger import get_logger
import pickle
from src.exception import CustomException
import sys
import json

logger = get_logger('Main_utils')

def save_object(object, file):

    try: 
        logger.info('Saving Object')

        with open(file, 'wb') as File: 
            pickle.dump(object, File)
    except Exception as e: 
        logger.error("Something Went Wrong while saving preprocessor")
        raise CustomException(e, sys)


import json
import os
from src.logger import get_logger
from src.exception import CustomException
import sys

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
