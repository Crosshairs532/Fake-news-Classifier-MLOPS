from src.logger import get_logger
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from keras.layers import LSTM, Dense
from keras.layers import Embedding
from keras import Sequential, Input
from src.exception import CustomException
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
from src.utils.main_utils import *
import os
import mlflow
import mlflow.keras
import dagshub

ml_flow_tracking_url = "https://dagshub.com/Crosshairs532/Fake-news-Classifier-MLOPS.mlflow/"

mlflow.set_tracking_uri(ml_flow_tracking_url)
dagshub.init(repo_owner='Crosshairs532', repo_name='Fake-news-Classifier-MLOPS', mlflow=True)

logger = get_logger('Model training')

class ModelTrainer: 
    def __init__(self, feature_engineering_artifact, df):
        self.feature_engineering_artifact = feature_engineering_artifact
        self.token_size = None
        self.df = df
        self.scores = None 
    
        

    def create_model(self):

        input = Input(shape=(8,))
        Embeddding = Embedding(input_dim=self.token_size, output_dim=10)
        lstm = LSTM(50)
        model  = Sequential()
        model.add(input)
        model.add(Embeddding)
        model.add(lstm)
        output = model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def model_object(self, x, y):
        logger.info(f"Padded: {type(x), len(x) } new_df:{y.shape}")

        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

            model = self.create_model()

            logger.info('Model training started')
            model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32)
            logger.info("Model Training Finished")


            y_pred = (model.predict(x_test) > 0.5).astype(dtype='int8')
            accuracy = accuracy_score(y_test, y_pred) 
            f1 = f1_score(y_test, y_pred)
            recall_score = recall_score(y_test, y_pred)
            precision_score =  precision_score(y_test, y_pred)

            self.scores = {
                "accuracy": accuracy,
                "f1": f1,
                "recall_score":recall_score,
                "precision_score":precision_score
            }

            return model

        except Exception as e: 
            raise CustomException(e, sys)
    

    def save_model_info(run_id: str, model_path: str, file_path: str) -> None:

        try:
            model_info = {'run_id': run_id, 'model_path': model_path}
            with open(file_path, 'w') as file:
                json.dump(model_info, file, indent=4)
            logger.debug('Model info saved to %s', file_path)
        except Exception as e:
            logger.error('Error occurred while saving the model info: %s', e)
            raise

    def initiate_model_trainer(self, x):
        

        logger.info("Model Training Stasrted")
        logger.info('loading feature config..')

        mlflow.set_experiment('Fake-News-Classifier')

        with mlflow.start_run() as run: 

            feature_config = load_config('artifacts/preprocessor/feature_config.json')
            self.token_size = feature_config['vocab_size']

            #log params
            mlflow.log_param('Vocab_size', self.token_size)
            mlflow.log_param('max_len', feature_config['max_len'])


            model = self.model_object(x, self.df['label'])
            model_dir = os.path.join("artifacts", "models")
            os.makedirs(model_dir, exist_ok=True)
            save_path = os.path.join(model_dir, 'model.pkl')
            save_object(model, save_path)


            mlflow.log_artifact(save_path)
            mlflow.log_artifact("artifacts/preprocessor/preprocessor.pkl")
            mlflow.log_artifact("artifacts/preprocessor/feature_config.json")


            mlflow.log_metric("accuracy", self.scores["accuracy"])
            mlflow.log_metric("f1_score", self.scores["f1"])
            mlflow.log_metric("precision", self.scores["precision"])
            mlflow.log_metric("recall", self.scores["recall"])

            mlflow.keras.log_model(model, "FakeNewsClassifier")


            self.save_model_info(run.info.run_id, "artifacts/models", 'reports/experiment_info.json')



            logger.info('Model Saved')
        
        


        

        
        