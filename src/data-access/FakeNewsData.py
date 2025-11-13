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
    
    def import_collection_as_dataframe(self, collection_name:str=None, database_name = None):
        try: 
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            

            logger.info("Fetching Data from mongoDb...")
            df = pd.DataFrame(collection.find())
            if "id" in df.columns.to_list():
                df = df.drop(columns=["id"], axis=1)
            df.replace({"na":np.nan},inplace=True)

            logger.info("Data Fetched Successfully")
            return df
        except Exception as e: 
            raise CustomException("Failed to Fetch Data!!", sys)
            