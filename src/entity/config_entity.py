
from dataclasses import dataclass
from src.constant import * 
import os

@dataclass
class TrainingPipelineConfig:
    artifact_dir: str = os.path.join(ARTIFACT_DIR)

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, feature_store)
    feature_store_file_path = os.path.join(data_ingestion_dir, feature_store_file_name)
    train_data_dir = os.path.join(training_pipeline_config.artifact_dir, 'train')
    test_data_dir = os.path.join(training_pipeline_config.artifact_dir, 'test')
    train_file_path = os.path.join(train_data_dir , 'train.csv')
    test_file_path = os.path.join(test_data_dir , 'test.csv')