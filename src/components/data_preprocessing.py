from src.logger import get_logger
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from src.entity.artifact_entity import DataPreProcessingArtifact
from src.entity.config_entity import DataCorpusConfig
import pandas as pd 
import numpy as np
import os
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = get_logger("Data preprocessing")

class DataPreProcessing:
    def __init__(self, dataIngestionArtifact):
        self.dataIngestionArtifact = dataIngestionArtifact
        self.data_corpus_config = DataCorpusConfig()
   
    def preprocess_data(self, df):
        df = df.dropna()
        ps = PorterStemmer()
        corpus = []
        for i in range(len(df)):
            review = re.sub('[^a-zA-Z]', ' ', df['title'].iloc[i])
            review = review.lower()
            review = review.split()
            review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
        return pd.DataFrame({
            'corpus':corpus, 
            'label':df['label']
        })

    def initiate_data_preprocessing(self):
        logger.info("Data preprocessing Started")

        train_df = pd.read_csv(self.dataIngestionArtifact.trained_file_path)
        test_df = pd.read_csv(self.dataIngestionArtifact.test_file_path)

        train_corpus= self.preprocess_data(train_df)
        test_corpus = self.preprocess_data(test_df)


        os.makedirs(self.data_corpus_config.data_corpus_dir, exist_ok=True)
        train_corpus.to_csv(self.data_corpus_config.train_corpus_file_path, index=False, header=True)
        test_corpus.to_csv(self.data_corpus_config.test_corpus_file_path, index=False, header=True)


        dataPreProcessingArtifact = DataPreProcessingArtifact(
            train_corpus_file_path=self.data_corpus_config.train_corpus_file_path,
            test_corpus_file_path=self.data_corpus_config.test_corpus_file_path
        )
        
        logger.info("Data preprocessing Finished")

        return dataPreProcessingArtifact

if __name__ == "__main__":
    from src.entity.config_entity import DataIngestionConfig
    from src.entity.artifact_entity import DataIngestionArtifact
    config = DataIngestionConfig()
    artifact = DataIngestionArtifact(trained_file_path=config.train_file_path, test_file_path=config.test_file_path)
    obj = DataPreProcessing(artifact)
    obj.initiate_data_preprocessing()




      
            


