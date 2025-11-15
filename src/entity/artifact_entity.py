from dataclasses import dataclass



@dataclass
class DataFeatureEngineerArtifact:
    preprocessor_file_path:str
    feature_config_file_path:str