import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")

class DataProcessor:
    def __init__(self) -> None:
        # Initialize stop words
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(list(ENGLISH_STOP_WORDS))
        self.stop_words = list(set(self.stop_words))
        self.stop_words = ' '.join(self.stop_words)
        self.stop_words = re.sub(r'[^a-zA-Z0-9\s]', ' ', self.stop_words) 
        self.stop_words = self.stop_words.split()
    
    def format_csv(self, filePath, columnsToRemove=[]):
        # Open the csv
        df = pd.read_csv(filePath)

        # Drop columns that have unnecessary data
        df.drop(columns=columnsToRemove, inplace=True)

        # Drop rows with null values
        df.dropna(inplace=True)

        return df


    def process_data(self, df):
        processed_quotes = []

        for index, row in df.iterrows():
            category = row['Category']
            quote = row['Quote']
            
            processed_text = self.process_text(quote)

            processed_quotes.append({'category': category, 'quote': processed_text})

        # Convert processed_quotes to a DataFrame
        processed_df = pd.DataFrame(processed_quotes)

        return processed_df
    
    def process_text(self, text):
        text  = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        words = word_tokenize(text)

        clean_words = []
        for word in words:
            if word not in self.stop_words:
                clean_words.append(word)

        return ' '.join(clean_words)