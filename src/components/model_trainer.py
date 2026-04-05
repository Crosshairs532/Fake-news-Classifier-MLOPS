from src.logger import get_logger
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras import Sequential, Input
from src.exception import CustomException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import sys
from src.utils.main_utils import *
import os
import mlflow
import mlflow.keras
import dagshub
import json
from dotenv import load_dotenv

load_dotenv()


dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Plain URI — no credentials embedded
mlflow.set_tracking_uri("https://dagshub.com/Crosshairs532/Fake-news-Classifier-MLOPS.mlflow")


tf_gpus = tf.config.list_physical_devices('GPU')
if tf_gpus:
    try:
        tf.config.set_visible_devices([], 'GPU')
        logger = get_logger('Model training')
        logger.info("Disabled tensorflow-metal GPU acceleration to avoid LSTM hanging bugs.")
    except RuntimeError as e:
        logger.error(e)
 
 
logger = get_logger('Model training')
 
logger.info(f"Checking Environment Variables...")
logger.info(f"MLFLOW_TRACKING_URI present: {bool(os.getenv('MLFLOW_TRACKING_URI'))}")
logger.info(f"MLFLOW_TRACKING_USERNAME present: {bool(os.getenv('MLFLOW_TRACKING_USERNAME'))}")
logger.info(f"MLFLOW_TRACKING_PASSWORD present: {bool(os.getenv('MLFLOW_TRACKING_PASSWORD'))}")
class ModelTrainer: 
    def __init__(self, feature_engineering_artifact):
        self.feature_engineering_artifact = feature_engineering_artifact
        self.token_size = None
        self.max_len = None
        self.scores = {}
 
    def create_model(self):
        # Architecture aligned with reference notebook:
        # Embedding(voc_size=5000, dim=40) -> Dropout(0.3) -> LSTM(100) -> Dropout(0.3) -> Dense(sigmoid)
        embedding_vector_features = 40
        model = Sequential([
            Input(shape=(self.max_len,)),
            Embedding(input_dim=self.token_size, output_dim=embedding_vector_features,
                      input_length=self.max_len),
            Dropout(0.3),
            LSTM(100),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
 
    def model_object(self, train_arr, test_arr):
        try:
            # x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            # x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
 
            # y_train = y_train.astype('int32')
            # y_test = y_test.astype('int32')
 
            # print("x_train.shape", x_train.shape)
            # print("y_train.shape", y_train.shape)
            # print("x_test.shape", x_test.shape)
            # print("y_test.shape", y_test.shape)
            x_train, x_test, y_train, y_test = train_test_split(train_arr[:, :-1], train_arr[:, -1], test_size=0.2, random_state=42)
            
            model = self.create_model()
 
            logger.info('Model training started')
            # batch_size=64, epochs=10 aligned with reference notebook
            model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
 
            y_pred = (model.predict(x_test) > 0.5).astype('int8')
 
            self.scores = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred)
            }
 
            return model
 
        except Exception as e: 
            raise CustomException(e, sys)
    
    @staticmethod
    def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
        os.makedirs('reports', exist_ok=True)
        try:
            model_info = {'run_id': run_id, 'model_path': model_path}
            with open(file_path, 'w') as file:
                json.dump(model_info, file, indent=4)
            logger.debug('Model info saved to %s', file_path)
        except Exception as e:
            logger.error('Error occurred while saving the model info: %s', e)
            raise
 
    def initiate_model_trainer(self, train_arr, test_arr):
        logger.info("Model Training Started")
        logger.info('loading feature config..')


        logger.info(f"TRACKING URI: {mlflow.get_tracking_uri()}")
        logger.info(f"USERNAME VALUE: {os.getenv('MLFLOW_TRACKING_USERNAME')}")
        logger.info(f"PASSWORD first 6: {str(os.getenv('MLFLOW_TRACKING_PASSWORD'))[:6]}")
 
        mlflow.set_experiment('Fake-News-Classifier-MLOPS-v1')
 
        with mlflow.start_run() as run: 
            feature_config = load_config('artifacts/preprocessor/feature_config.json')
            self.token_size = feature_config['vocab_size']
            self.max_len = feature_config['max_len']
 
            # log params
            mlflow.log_param('Vocab_size', self.token_size)
            mlflow.log_param('max_len', self.max_len)
 
            model = self.model_object(train_arr, test_arr)


            model_data = {
                "architecture": model.to_json(),
                "weights": model.get_weights()
            }

            model_dir = os.path.join("artifacts", "models")
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, 'model.pkl')
            save_object(model_data, save_path)
 
            mlflow.log_artifact(save_path)
            # mlflow.log_artifact("artifacts/preprocessor/preprocessor.pkl")
            mlflow.log_artifact("artifacts/preprocessor/feature_config.json")
 
            mlflow.log_metric("accuracy", self.scores["accuracy"])
            mlflow.log_metric("f1_score", self.scores["f1"])
            mlflow.log_metric("precision", self.scores["precision"])
            mlflow.log_metric("recall", self.scores["recall"])
 
            mlflow.keras.log_model(model, "FakeNewsClassifier")
            self.save_model_info(run.info.run_id, "artifacts/models",   "reports/experiment_info.json" )
 
            logger.info('Model Saved')
 
 
if __name__ == "__main__":
    from src.entity.config_entity import DataFeatureEngineeringConfig
    from src.entity.artifact_entity import DataFeatureEngineerArtifact
    from src.utils.main_utils import load_numpy_array_data
    import os
    
    config = DataFeatureEngineeringConfig()
    artifact = DataFeatureEngineerArtifact(
        preprocessor_file_path=config.preprocessor_file_path,
        feature_config_file_path=config.feature_config_file_path
    )
    
    train_arr = load_numpy_array_data(config.train_arr_file_path)
    test_arr = load_numpy_array_data(config.test_arr_file_path)
    
    trainer = ModelTrainer(artifact)
    trainer.initiate_model_trainer(train_arr, test_arr)