from dotenv import load_dotenv
import os 
from src.logger import get_logger
from src.constant import * 
from src.exception import CustomException
from pymongo import MongoClient
import sys
from typing import Optional
import pandas as pd

load_dotenv()
logger = get_logger('Mongodb_connection')

mongodb_username=os.getenv('DB_NAME')
mongodb_password=os.getenv('DB_PASSWORD')

class MongoDBClient:

    client = None
    inserted = False

    def __init__(self, database_name:Optional[str] = MONGODB_DATABASE_NAME ):
        logger.info("Connecting to Mongodb...")
        test_collection_name = "test-collection"  # Change to your main collection
        submit_collection_name = "submit-collection"  # Change to your main collection
        try: 
            if  MongoDBClient.client is None: 
                if not mongodb_username or not mongodb_password:
                    raise CustomException("Mongodb username/password not found!", sys)  
                mongodb_uri = "mongodb+srv://admin:KdqbhZulPlXQNa1O@ml-clusters.qazmdxn.mongodb.net/fake_news?retryWrites=true&w=majority&appName=Ml-clusters"

                MongoDBClient.client = MongoClient(mongodb_uri)

            self.client = MongoDBClient.client
        
            self.database = self.client[database_name]  

            if not MongoDBClient.inserted:


                test_collection = self.database[test_collection_name]
                submit_collection = self.database[submit_collection_name]

                if test_collection.count_documents({}) == 0:
                    df1  = pd.read_csv('Notebooks/dataset/test.csv', sep=',', engine='python')
                    df2  = pd.read_csv('Notebooks/dataset/submit.csv', sep=',', engine='python')
                    test_collection.insert_many(df1.to_dict(orient='records'))
                    submit_collection.insert_many(df2.to_dict(orient='records'))

                    MongoDBClient.inserted= True

                    logger.info(f"Inserted document into '{test_collection_name}' collection.")
                    logger.info(f"Inserted document into '{submit_collection_name}' collection.")

                self.submit_collection = self.database[submit_collection_name]
                self.test_collection = self.database[test_collection_name]

            logger.info("Connected to mongodb.")
        except Exception as e: 
            raise CustomException("Failed to connect to mongodb!!", sys)