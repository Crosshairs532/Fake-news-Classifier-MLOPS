import warnings
warnings.filterwarnings("ignore")
from src.data_access.FakeNewsData import FakeNewsData
from src.logger import get_logger
from sklearn.metrics import accuracy_score, f1_score
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.sequence import pad_sequences

from mlflow import log_metric, log_param, log_params, log_artifacts
import mlflow
import dagshub

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreProcessing
from src.components.data_feature_engineering import DataFeatureEngineering
from src.components.model_trainer import ModelTrainer

logger = get_logger("Demo")

if __name__ =="__main__":
    data_ingestion = DataIngestion()
    data_preprocessing = DataPreProcessing()
    data_feature_engineering = DataFeatureEngineering(maxlen=8)

    df = data_ingestion.initialize_data_ingestion()
    corpus, new_df = data_preprocessing.initiate_data_preprocessing(df)
    padded, feature_engineering_artifact  = data_feature_engineering.initialize_feature_engineering(corpus)

    model_trainer = ModelTrainer(feature_engineering_artifact, new_df)

    model_trainer.initiate_model_trainer(padded)
    


