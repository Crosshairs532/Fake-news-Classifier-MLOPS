
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

@dataclass
class DataCorpusConfig:
    data_corpus_dir = os.path.join(training_pipeline_config.artifact_dir, 'data_corpus')
    train_corpus_file_path = os.path.join(data_corpus_dir, 'train_corpus.csv')
    test_corpus_file_path = os.path.join(data_corpus_dir, 'test_corpus.csv')

@dataclass
class DataFeatureEngineeringConfig:
    preprocessor_file_path = os.path.join(training_pipeline_config.artifact_dir, preprocessor, 'preprocessor.pkl')
    feature_config_file_path = os.path.join(training_pipeline_config.artifact_dir, preprocessor, "feature_config.json")
    train_arr_file_path = os.path.join(training_pipeline_config.artifact_dir, preprocessor, 'train.npz')
    test_arr_file_path = os.path.join(training_pipeline_config.artifact_dir, preprocessor, 'test.npz')
