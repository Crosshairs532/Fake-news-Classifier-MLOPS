from dotenv import load_dotenv
import os 
from src.logger import get_logger
from src.constant import * 
from src.exception import CustomException
from pymongo import MongoClient
import sys

logger = get_logger('Mongodb_connection')

mongodb_username=os.getenv('DB_NAME')
mongodb_password=os.getenv('DB_PASSWORD')

class MongoDBClient:
    client = None
    
    def __init__(self, database_name:str = MONGODB_DATABASE_NAME ):
        logger.info("Connecting to Mongodb...")
        try: 
            if MongoDBClient.client is None: 
                if not mongodb_username or not mongodb_password:
                    raise CustomException("Mongodb username/password not found!")
                
                mongodb_uri = f"mongodb+srv://{mongodb_username}:{mongodb_password}@ml-clusters.qazmdxn.mongodb.net/fake-news?appName=Ml-clusters"
                MongoDBClient.client = MongoClient(mongodb_uri)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]  
            self.database_name = database_name

            logger.info("Connected to mongodb.")
        except Exception as e: 
            raise CustomException("Failed to connect to mongodb!!", sys)