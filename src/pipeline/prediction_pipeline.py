import os
import sys
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import tensorflow as tf
from src.logger import get_logger
from src.exception import CustomException
from src.utils.main_utils import load_config

logger = get_logger("PredictionPipeline")

class PredictionPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "models", "model.pkl")
        self.config_path = os.path.join("artifacts", "preprocessor", "feature_config.json")

    def preprocess_text(self, text: str):
        try:
            logger.info("Preprocessing input text")
            # Try to download stopwords if not already available
            try:
                stopwords.words('english')
            except LookupError:
                nltk.download('stopwords')
                
            ps = PorterStemmer()
            review = re.sub('[^a-zA-Z]', ' ', text)
            review = review.lower()
            review = review.split()
            review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
            review = ' '.join(review)
            return [review]
        except Exception as e:
            raise CustomException(e, sys)

    def load_object(self, file_path: str):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading object from {file_path}")
            raise CustomException(e, sys)

    def predict(self, text: str):
        try:
            logger.info("Starting prediction process")

            # 1. Preprocess the raw text
            processed_text = self.preprocess_text(text)

            # 2. Load the Preprocessor (Tokenizer)
            tokenizer = self.load_object(self.preprocessor_path)

            # 3. Load configuration
            config = load_config(self.config_path)
            max_len = config['max_len']

            # 4. Tokenization and Padding
            seq = tokenizer.texts_to_sequences(processed_text)
            padded = tf.keras.utils.pad_sequences(
                seq,
                maxlen=max_len,
                dtype='int32',
                padding='pre',
                truncating='pre',
                value=0.0
            )

            # 5. Load the trained Model
            model = self.load_object(self.model_path)

            # 6. Predict the output
            prediction = model.predict(padded)
            # Binary classification (0 = Real, 1 = Fake) generally depends on training labels
            is_fake = (prediction[0][0] > 0.5).astype(dtype='int8')
            
            logger.info("Prediction successful")
            return int(is_fake)

        except Exception as e:
            raise CustomException(e, sys)
