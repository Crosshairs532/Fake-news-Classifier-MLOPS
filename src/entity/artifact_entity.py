from dataclasses import dataclass
from typing import Any


@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str

@dataclass
class DataPreProcessingArtifact:
    train_corpus_file_path:str
    test_corpus_file_path:str    

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class DataFeatureEngineerArtifact:
    preprocessor_file_path: str
    feature_config_file_path: str



@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float
    auc_roc:float
    accuracy:float

@dataclass
class ModelTrainerArtifact:
    model_path: str
    metric_artifact:ClassificationMetricArtifact

