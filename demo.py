
from src.data_access.FakeNewsData import FakeNewsData
from src.logger import get_logger
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")
import nltk
import numpy as np 
import pandas as pd
import regex as re
import tensorflow
from tensorflow import keras
from nltk.corpus import stopwords
from keras.layers import Embedding
from keras import Sequential, Input
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Dense
from mlflow import log_metric, log_param, log_params, log_artifacts
import mlflow
import dagshub

from src.components.data_ingestion import DataIngestion




logger = get_logger("Demo")
if __name__ =="__main__":
    data = DataIngestion()
    data.initialize_data_ingestion()

    # data = FakeNewsData().import_collection_as_dataframe('test-collection', "submit-collection")
    # data.dropna(inplace=True)
    # X = data.drop(['label'], axis = 1)
    # y = data['label']

    # print(X.shape, y.shape)

    # ps = PorterStemmer()
    # corpus = []
    # for i in range(0, len(X)):
    #     review = re.sub('[^a-zA-Z]', ' ',  str(X['title'].iloc[i]))
    #     review = review.lower()
    #     review = review.split()
        
    #     review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    #     review = ' '.join(review)
    #     corpus.append(review)
    # # OHE 
    # tokenize  = Tokenizer()
    # tokenize.fit_on_texts(corpus)

    # OHE_representation = tokenize.texts_to_sequences(corpus)

    # # Padding 
    # padded = tensorflow.keras.utils.pad_sequences(
    #     OHE_representation,
    #     maxlen=8,
    #     dtype='int32',
    #     padding='pre',
    #     truncating='pre',
    #     value=0.0
    # )

    # all_words = []

    # for  sentence in corpus:
    #     words =  nltk.tokenize.word_tokenize(sentence) 
    #     all_words.extend(words)
        
    # tokens_size = len(set(all_words))
    # input = Input(shape=(8,))
    # Embeddding = Embedding(input_dim=tokens_size, output_dim=10)
    # lstm = LSTM(50)

    # model  = Sequential()

    # model.add(input)
    # model.add(Embeddding)
    # model.add(lstm)
    # output = model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # # split Data
    # from  sklearn.model_selection import train_test_split
    # ml_flow_tracking_url = "https://dagshub.com/Crosshairs532/Fake-news-Classifier-MLOPS.mlflow/"

    # mlflow.set_tracking_uri(ml_flow_tracking_url)
    # dagshub.init(repo_owner='Crosshairs532', repo_name='Fake-news-Classifier-MLOPS', mlflow=True)

    # mlflow.set_experiment('Fake-news')
    # with mlflow.start_run() as run:

    #     logger.info(f"X: {len(np.array(padded))}, y: {len(np.array(y))}")    
    #     X_train, X_test, y_train, y_test = train_test_split(np.array(padded), np.array(y), test_size=0.33, random_state=1)            
        
    #     logger.info(f"X_train type: {type(X_train)}")
    #     model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    #     y_pred = (model.predict(X_test) > 0.5).astype(dtype='int8')
        
    #     accuracy = accuracy_score(y_test, y_pred)
    #     f1_score_ = f1_score(y_test, y_pred)

    #     mlflow.log_metrics("metric",{
    #         'accuracy':accuracy,
    #         'f1_score':f1_score_
    #     })
    #     mlflow.log_params('params',{
    #         'unit':100,
    #         'token-size':tokens_size,
    #     })
    #     mlflow.keras.log_model(model, "LSTM")


