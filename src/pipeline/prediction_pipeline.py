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
import dagshub
import mlflow
import os 
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

logger = get_logger("PredictionPipeline")


## Dagshub Setup
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
dagshub.init(repo_owner='Crosshairs532', repo_name='Fake-news-Classifier-MLOPS', mlflow=True)

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

    def get_latest_model(model_name):
        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        return latest_version[0].version if latest_version else None

    def predict(self, text: str):
        try:
            logger.info("Starting prediction process")
            # Preprocess the raw text
            processed_text = self.preprocess_text(text)

            # Load the Preprocessor (Tokenizer)
            tokenizer = self.load_object(self.preprocessor_path)

            # Load configuration
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

            # Load the trained Model
            # model_data = self.load_object(self.model_path)
            # model = tf.keras.models.model_from_json(model_data["architecture"])
            # model.set_weights(model_data["weights"])
            # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model_name = "FakeNewsClassifier"
            model = self.get_latest_model(model_name)
            model_uri = f'models:/{model_name}/{model.model_version}'
            model = mlflow.pyfunc.load_model(model_uri)

            # predict
            prediction = model.predict(padded)

            is_fake = (prediction[0][0] > 0.5).astype(dtype='int8')
            
            logger.info("Prediction successful")
            return int(is_fake)

        except Exception as e:
            raise CustomException(e, sys)
