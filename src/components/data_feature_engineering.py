from tensorflow.keras.preprocessing.text import Tokenizer
from src.entity.artifact_entity import DataFeatureEngineerArtifact
from src.utils.main_utils import * 
import numpy as np
import pandas as pd
import tensorflow
import nltk
import json
import os




class DataFeatureEngineering: 
    def __init__(self, data_featureEngineering_config, data_preprocessing_artifact = None):
        self.tokenizer = Tokenizer()
        self.token_size = None
        self.max_len = 8
        self.train_df = pd.read_csv(data_preprocessing_artifact.train_corpus_file_path)
        self.test_df = pd.read_csv(data_preprocessing_artifact.test_corpus_file_path)
        self.data_featureEngineering_config = data_featureEngineering_config
        self.data_preprocessing_artifact = data_preprocessing_artifact

        os.makedirs(
            os.path.dirname(self.data_featureEngineering_config.preprocessor_file_path), 
            exist_ok=True
        )


    def fit(self, corpus):
        self.tokenizer.fit_on_texts(corpus)
        
        # Keras Tokenizer starts indices at 1 and 0 is reserved for padding, so input_dim needs to be len(word_index) + 1
        self.token_size = len(self.tokenizer.word_index) + 1

        os.makedirs(os.path.dirname(self.data_featureEngineering_config.preprocessor_file_path), exist_ok=True)
        save_object(self.tokenizer, file=self.data_featureEngineering_config.preprocessor_file_path)

        config = {
            "vocab_size": self.token_size,
            "max_len": self.max_len
        }

        with open(self.data_featureEngineering_config.feature_config_file_path, "w") as f:
            json.dump(config, f, indent=4)


        return self.token_size, self.data_featureEngineering_config.feature_config_file_path
        
    def transform(self, corpus):

        OHE_representation = self.tokenizer.texts_to_sequences(corpus)

        padded = tensorflow.keras.utils.pad_sequences(
            OHE_representation,
            maxlen=8,
            dtype='int32',
            padding='pre',
            truncating='pre',
            value=0.0
        )

        return padded

    
    def initialize_feature_engineering(self):
        logger.info("Feature Engineering Started.")
        try:

            train_df = pd.read_csv(self.data_preprocessing_artifact.train_corpus_file_path)
            test_df = pd.read_csv(self.data_preprocessing_artifact.test_corpus_file_path)

            train_corpus = train_df['corpus'].astype(str).tolist()
            vocab_size, config_path = self.fit(train_corpus)
            test_corpus = test_df['corpus'].astype(str).tolist()

            X_train_arr = self.transform(train_corpus)
            X_test_arr = self.transform(test_corpus)

            # label
            y_train = train_df['label'].values
            y_test = test_df['label'].values

            train_arr = np.c_[X_train_arr, y_train]
            test_arr = np.c_[X_test_arr, y_test]





            save_numpy_array_data(file_path=os.path.join(os.path.dirname(self.data_preprocessing_artifact.train_corpus_file_path), "train.npz"), array=train_arr)
            save_numpy_array_data(file_path=os.path.join(os.path.dirname(self.data_preprocessing_artifact.test_corpus_file_path), "test.npz"), array=test_arr)

            preprocess_artifact = DataFeatureEngineerArtifact(
                preprocessor_file_path = self.data_featureEngineering_config.preprocessor_file_path,
                feature_config_file_path = self.data_featureEngineering_config.feature_config_file_path,
            )
            return train_arr, test_arr, preprocess_artifact
        except Exception as e: 
            raise CustomException(e, sys)

