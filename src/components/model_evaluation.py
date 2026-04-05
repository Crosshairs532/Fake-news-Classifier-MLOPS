import dagshub
from src.exception import CustomException
from src.logger import get_logger
from dotenv import load_dotenv
from src.utils.main_utils import load_config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.main_utils import load_numpy_array_data
from src.entity.config_entity import DataFeatureEngineeringConfig
import mlflow.keras
import tensorflow as tf
import mlflow
import dagshub
import sys
import os
import json
import pickle

load_dotenv()

logger = get_logger("ModelEvaluation")
dagshub_token = os.getenv("CAPSTONE_TEST")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri("https://dagshub.com/Crosshairs532/Fake-news-Classifier-MLOPS.mlflow")

class ModelEvaluation: 
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "models", "model.pkl")
        self.config_path = os.path.join("artifacts", "preprocessor", "feature_config.json")
    def load_object(self, file_path: str):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading object from {file_path}")
            raise CustomException(e, sys)
    def initiate_model_evaluation(self, test_arr):
        logger.info("Model Evaluation Started.")

        mlflow.set_experiment("model-evaluation")
        with mlflow.start_run() as run:
            try:
                X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
                tokenizer = self.load_object(self.preprocessor_path)
                config = load_config(self.config_path)
                max_len = config['max_len']
                model_data = self.load_object(self.model_path)
                arch_dict = model_data["architecture"]



                # Load the trained Model
                model_data = self.load_object(self.model_path)
                model = tf.keras.models.model_from_json(model_data["architecture"])
                model.set_weights(model_data["weights"])
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

                # predict
                y_pred_probs = model.predict(X_test)
                y_pred = (y_pred_probs > 0.5).astype("int32")

                acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.keras.log_model(model, name="FakeNewsClassifier")

                logger.info(f"model Evaludation Completed\n Accuracy: {acc}\n Precision: {precision}\n Recall: {recall}\n F1 Score: {f1}")

            except Exception as e:
                raise CustomException(e, sys)

if __name__ == "__main__":
    """
        1. data
        2. model.pkl
        3. preprocessor.pkl
        4. feature_config.json
        5. experiment_info.json
    """
    
    model_evaluation = ModelEvaluation()
    config = DataFeatureEngineeringConfig()
    test_arr = load_numpy_array_data(config.test_arr_file_path)
    model_evaluation.initiate_model_evaluation(test_arr)