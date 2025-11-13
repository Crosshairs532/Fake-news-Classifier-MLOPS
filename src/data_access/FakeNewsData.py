from src.logger import get_logger
from src.configuration.mongodb_connection import MongoDBClient
from src.exception import CustomException
import sys
import pandas as pd
import numpy as np

logger = get_logger("Fake news Data")

class FakeNewsData:
    def __init__(self):
        try: 
            self.client = MongoDBClient()
            
        except Exception as e: 
            raise CustomException(e, sys)
    
    def import_collection_as_dataframe(self, collection_name1:str=None,collection_name2=None, database_name = None):
        try: 
            if database_name is None:
                test_collection = self.client.database[collection_name1]
                submit_collection = self.client.database[collection_name2]
            else:
                test_collection = self.client[database_name][test_collection]
                submit_collection = self.client[database_name][submit_collection]
            

            logger.info("Fetching Data from mongoDb...")
            test = pd.DataFrame(test_collection.find())
            submit = pd.DataFrame(submit_collection.find())

            X = test.copy()
            y = submit.copy()

            new_df = pd.merge(X, y, how='left', on='id')
            new_df.replace({"na":np.nan},inplace=True)

            logger.info("Data Fetched Successfully")
            return new_df
        except Exception as e: 
            raise CustomException("Failed to Fetch Data!!", sys)
            