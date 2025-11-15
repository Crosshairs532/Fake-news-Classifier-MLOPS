from tensorflow.keras.preprocessing.text import Tokenizer
from src.entity.artifact_entity import DataFeatureEngineerArtifact
import tensorflow
import nltk
import os
from src.utils.main_utils import * 
import json

class DataFeatureEngineering: 
    def __init__(self, maxlen = 8):
        self.tokenizer = None
        self.token_size = None
        self.max_len = maxlen

        os.makedirs('artifacts/preprocessor', exist_ok=True)


    def fit(self, corpus):
        all_words = []
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(corpus)
        for  sentence in corpus:
            words =  nltk.tokenize.word_tokenize(sentence) 
            all_words.extend(words)
        self.token_size = len(set(all_words))

        preprocessor_dir = os.path.join("artifacts", "preprocessor")
        os.makedirs(preprocessor_dir, exist_ok=True)
        save_object(self.tokenizer, file=os.path.join(preprocessor_dir, 'preprocessor.pkl'))

        config = {
            "vocab_size": self.token_size,
            "max_len": self.max_len
        }

        with open(os.path.join('artifacts/preprocessor', "feature_config.json"), "w") as f:
            json.dump(config, f, indent=4)


        preprocess_artifact = DataFeatureEngineerArtifact(
            preprocessor_file_path = os.path.join('artifacts/preprocessor', 'preprocessor.pkl'),
            feature_config_file_path = os.path.join('artifacts/preprocessor', "feature_config.json")
        )
        return preprocess_artifact
        
        
        
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

    def fit_transform(self, corpus):
        feature_engineering_artifact = self.fit(corpus)

        return self.transform(corpus), feature_engineering_artifact
    
    def initialize_feature_engineering(self, corpus):

        
        logger.info("Feature Engineering Started.")
        logger.info(f"corpus: {len(corpus)}")

        try: 
            padded, feature_engineering_artifact = self.fit_transform(corpus)

            return padded, feature_engineering_artifact
        except Exception as e: 
            raise CustomException(e, sys)




