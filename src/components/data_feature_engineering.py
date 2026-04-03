from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.entity.artifact_entity import DataFeatureEngineerArtifact
from src.utils.main_utils import * 
import numpy as np
import pandas as pd
import json
import os


# Aligned with reference notebook (FakeNewsClassifierUsingLSTM.ipynb)
VOC_SIZE = 5000   # fixed hash vocabulary size used by one_hot
SENT_LENGTH = 20  # max sequence length (was 8 — changed to match reference)


class DataFeatureEngineering: 
    def __init__(self, data_featureEngineering_config, data_preprocessing_artifact=None):
        self.voc_size = VOC_SIZE
        self.sent_length = SENT_LENGTH
        self.train_df = pd.read_csv(data_preprocessing_artifact.train_corpus_file_path)
        self.test_df = pd.read_csv(data_preprocessing_artifact.test_corpus_file_path)
        self.data_featureEngineering_config = data_featureEngineering_config
        self.data_preprocessing_artifact = data_preprocessing_artifact

        os.makedirs(
            os.path.dirname(self.data_featureEngineering_config.preprocessor_file_path), 
            exist_ok=True
        )

    def fit(self, corpus):

        os.makedirs(os.path.dirname(self.data_featureEngineering_config.preprocessor_file_path), exist_ok=True)

        config = {
            "vocab_size": self.voc_size,
            "max_len": self.sent_length
        }

        with open(self.data_featureEngineering_config.feature_config_file_path, "w") as f:
            json.dump(config, f, indent=4)

        logger.info(f"Feature config saved — vocab_size={self.voc_size}, max_len={self.sent_length}")
        return self.voc_size, self.data_featureEngineering_config.feature_config_file_path

    def transform(self, corpus):
        onehot_repr = [one_hot(words, self.voc_size) for words in corpus]
        padded = pad_sequences(
            onehot_repr,
            maxlen=self.sent_length,
            padding='pre',
            truncating='pre',
        )
        return padded

    def initialize_feature_engineering(self):
        logger.info("Feature Engineering Started.")
        try:
            train_df = pd.read_csv(self.data_preprocessing_artifact.train_corpus_file_path)
            test_df = pd.read_csv(self.data_preprocessing_artifact.test_corpus_file_path)

            train_corpus = train_df['corpus'].astype(str).tolist()
            test_corpus = test_df['corpus'].astype(str).tolist()

            vocab_size, config_path = self.fit(train_corpus)

            X_train_arr = self.transform(train_corpus)
            X_test_arr = self.transform(test_corpus)

            # labels
            y_train = train_df['label'].values
            y_test = test_df['label'].values

            train_arr = np.c_[X_train_arr, y_train]
            test_arr = np.c_[X_test_arr, y_test]


            save_numpy_array_data(file_path=self.data_featureEngineering_config.train_arr_file_path, array=train_arr)
            save_numpy_array_data(file_path=self.data_featureEngineering_config.test_arr_file_path, array=test_arr)

            preprocess_artifact = DataFeatureEngineerArtifact(
                preprocessor_file_path=self.data_featureEngineering_config.preprocessor_file_path,
                feature_config_file_path=self.data_featureEngineering_config.feature_config_file_path,
            )
            return train_arr, test_arr, preprocess_artifact
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.entity.config_entity import DataCorpusConfig, DataFeatureEngineeringConfig
    from src.entity.artifact_entity import DataPreProcessingArtifact

    corpus_config = DataCorpusConfig()
    preprocessing_artifact = DataPreProcessingArtifact(
        train_corpus_file_path=corpus_config.train_corpus_file_path,
        test_corpus_file_path=corpus_config.test_corpus_file_path
    )
    feature_config = DataFeatureEngineeringConfig()
    obj = DataFeatureEngineering(data_featureEngineering_config=feature_config, data_preprocessing_artifact=preprocessing_artifact)
    obj.initialize_feature_engineering()