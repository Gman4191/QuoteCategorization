import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.feature_selection import SelectKBest, chi2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

class DataProcessor:
    def __init__(self) -> None:
        # Initialize stop words
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(list(ENGLISH_STOP_WORDS))
        self.stop_words = list(set(self.stop_words))
        self.stop_words = ' '.join(self.stop_words)
        self.stop_words = re.sub(r'-', ' ', self.stop_words)
        self.stop_words = re.sub(r'[^a-zA-Z0-9\s]', '', self.stop_words) 
        self.stop_words = self.stop_words.split()
        
        # Initialize lemmatization model
        self.nlp = spacy.load("en_core_web_sm")
    
    def format_csv(self, filePath, columnsToRemove):
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

        tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=.9)
        features = tfidf_vectorizer.fit_transform(processed_df['quote'])

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(processed_df['category'])

        k = len(set(encoded_labels)) * 3
        print(len(tfidf_vectorizer.get_feature_names_out()), "/", len(set(encoded_labels)), "=", k)

        selector = SelectKBest(score_func=chi2, k=k)
        tf_idf_selected = selector.fit_transform(features, encoded_labels)

        scaler = MaxAbsScaler()
        features = scaler.fit_transform(tf_idf_selected)

        return features, encoded_labels, processed_df
    
    def process_text(self, text):
        text  = text.lower()
        text = re.sub(r'-', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        words = word_tokenize(text)

        clean_words = []
        for word in words:
            if word not in self.stop_words:
                # Lemmatize using spaCy
                lemma = self.nlp(word)[0].lemma_
                clean_words.append(lemma)

        return ' '.join(clean_words)

# data_processor = DataProcessor()
# df = data_processor.format_csv('data/data.csv', [])
# features, encoded_labels, processed_df = data_processor.process_data(df.head(10))
