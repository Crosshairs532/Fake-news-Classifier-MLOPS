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

        mlflow.set_experiment("Fake-News-Classifier-MLOPS-v3")

        with mlflow.start_run() as run:

            # --- Load and evaluate (safe to wrap) ---
            try:
                X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
                config = load_config(self.config_path)

                model_data = self.load_object(self.model_path)
                model = tf.keras.models.model_from_json(model_data["architecture"])
                model.set_weights(model_data["weights"])
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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

                logger.info(
                    f"Model Evaluation Completed\n"
                    f" Accuracy:  {acc}\n"
                    f" Precision: {precision}\n"
                    f" Recall:    {recall}\n"
                    f" F1 Score:  {f1}"
                )

            except Exception as e:
                raise CustomException(e, sys)

            # --- Log model OUTSIDE try/except so failures are loud ---
            logger.info(f"Logging Keras model to MLflow run: {run.info.run_id}")
            mlflow.keras.log_model(model, name="FakeNewsClassifier")
            logger.info("log_model succeeded.")

            # --- Write experiment_info ONLY after log_model succeeds ---
            os.makedirs("reports", exist_ok=True)
            model_info = {
                "run_id": run.info.run_id,
                "model_path": "FakeNewsClassifier"
            }
            with open("reports/experiment_info.json", "w") as f:
                json.dump(model_info, f, indent=4)
            logger.info(f"experiment_info.json written with run_id: {run.info.run_id}")


if __name__ == "__main__":
    model_evaluation = ModelEvaluation()
    config = DataFeatureEngineeringConfig()
    test_arr = load_numpy_array_data(config.test_arr_file_path)
    model_evaluation.initiate_model_evaluation(test_arr)