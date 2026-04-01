import sys
from src.logger import get_logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreProcessing
from src.components.data_feature_engineering import DataFeatureEngineering
from src.entity.config_entity import *
from src.components.model_trainer import ModelTrainer
from src.components.register_model import register_model
import json
import numpy as np 
import pandas as pd

logger = get_logger("TrainPipeline")

class TrainPipeline:
    def __init__(self):
        self.data_featureEngineering_config = DataFeatureEngineeringConfig()

    def startDataIngestion(self):
        data_ingestion = DataIngestion()
        data_ingestion_artifact = data_ingestion.initialize_data_ingestion()
        return data_ingestion_artifact

    def startDataPreProcessing(self, dataIngestionArtifact):
        data_preprocessing = DataPreProcessing(dataIngestionArtifact)
        data_preprocessing_artifact = data_preprocessing.initiate_data_preprocessing()
        return data_preprocessing_artifact
    
    def startDataFeatureEngineering(self, dataPreprocessingArtifact):
        data_feature_engineering = DataFeatureEngineering(data_featureEngineering_config=self.data_featureEngineering_config, data_preprocessing_artifact=dataPreprocessingArtifact)
        return data_feature_engineering.initialize_feature_engineering()

    def run_pipeline(self):
        try:
            logger.info("Training pipeline execution started")

            # Data Ingestion
            data_ingestion_artifact = self.startDataIngestion()

            # Data Preprocessing
            data_preprocessing_artifact = self.startDataPreProcessing(data_ingestion_artifact)

            # Data Feature Engineering
            train_arr, test_arr, data_feature_engineering_artifact = self.startDataFeatureEngineering(data_preprocessing_artifact)


            # Model Training
            model_trainer = ModelTrainer(feature_engineering_artifact)
            model_trainer.initiate_model_trainer(train_arr, test_arr)

            # Model Registration
            logger.info("Fetching experiment info to register the model")
            try:
                with open('reports/experiment_info.json', 'r') as file:
                    model_info = json.load(file)
                
                register_model(model_name="FakeNewsClassifier", model_info=model_info)
                logger.info("Model Registration Completed successfully")
            except Exception as e:
                logger.error(f"Failed to register model due to {e}")

            logger.info("Training pipeline execution finished")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
