from src.logger import get_logger
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

logger = get_logger("Data preprocessing")

class DataPreProcessing:
    def __init__(self):
        pass
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
        return corpus, df

    def initiate_data_preprocessing(self, df):
        logger.info("Data preprocessing Started")

        print(df.head(4))
        corpus, new_df = self.preprocess_data(df)

        logger.info("Data preprocessing Finished")

        return corpus, new_df




      
            


