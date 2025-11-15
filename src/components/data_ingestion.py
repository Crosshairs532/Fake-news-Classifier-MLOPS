from src.logger import get_logger
from src.exception import CustomException
from src.data_access.FakeNewsData import FakeNewsData
from src.entity.config_entity import DataIngestionConfig
from sklearn.model_selection import train_test_split
import sys
import os
logger = get_logger('Data Ingestion')

class DataIngestion:
    def __init__(self):
        self.data_ingestion = DataIngestionConfig()
    

    def export_to_feature_store(self):
        try:
            logger.info("Importing Data from mongodb")

            data = FakeNewsData()
            df = data.import_collection_as_dataframe('test-collection', "submit-collection")
            if df is None: 
                logger.info("No Data found!")
                raise CustomException("No Data found!", sys)
            logger.info(f"Shape of data : {df.shape}")

            os.makedirs(self.data_ingestion.data_ingestion_dir, exist_ok=True)
            logger.info(f"Saving Whole data to feature store") 
            df.to_csv(self.data_ingestion.feature_store_file_path, index=False, header=True)
            logger.info(f"Data Saved to feature_store: {self.data_ingestion.feature_store_file_path}")
            return df

        except Exception as e: 
            raise CustomException("Failed to load data!!")
        
    def split_train_test(self, dataframe):

        try: 
            logger.info('Splitting data into train and test started')
            train, test = train_test_split(dataframe, random_state=1, test_size=0.3)
            logger.info("Data splitting Done")

            logger.info("Saving Data to train and test folder")

            os.makedirs(self.data_ingestion.train_data_dir, exist_ok=True)
            os.makedirs(self.data_ingestion.test_data_dir, exist_ok=True)


            train.to_csv(self.data_ingestion.train_file_path, index=True, header=True)
            test.to_csv(self.data_ingestion.test_file_path, index=True, header=True)


            logger.info(f"Train and test data saved: \nTrain path: {self.data_ingestion.train_file_path}\n Test path: {self.data_ingestion.test_file_path}")

            
        except Exception as e: 
            logger.error("Failed to Split Data")
            raise CustomException("Failed to Split Data", sys)

    def initialize_data_ingestion(self):
        logger.info("Data Ingestion started...")
        dataframe = self.export_to_feature_store()
        self.split_train_test(dataframe)
        logger.info("Exiting Data Ingestion")

        return dataframe

        



