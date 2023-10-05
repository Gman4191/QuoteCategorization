import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")

class DataProcessor:
    def __init__(self) -> None:
        """
        Initializes a DataProcessor object.
        
        This class is designed to process and clean textual data from CSV files. 
        It performs tasks such as removing stop words, dropping unnecessary 
        columns, and lowercasing text data.
        """

        # Initialize stop words
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(list(ENGLISH_STOP_WORDS))
        self.stop_words = list(set(self.stop_words))
        self.stop_words = ' '.join(self.stop_words)
        self.stop_words = re.sub(r'[^a-zA-Z0-9\s]', ' ', self.stop_words) 
        self.stop_words = self.stop_words.split()
    
    def format_csv(self, filePath, columnsToRemove=[]):
        """
        Reads a dataset CSV file and formats it by dropping specified columns and removing rows with null values.

        Parameters:
        - filePath (str): The path to the dataset CSV file to be processed.
        - columnsToRemove (list): A list of column names to be removed from the DataFrame.

        Returns:
        - df (DataFrame): The processed DataFrame.
        """

        # Open the csv
        df = pd.read_csv(filePath)

        # Drop columns that have unnecessary data
        df.drop(columns=columnsToRemove, inplace=True)

        # Drop rows with null values
        df.dropna(inplace=True)

        return df


    def process_data(self, df):
        """
        Processes the data in a DataFrame by lowercasing text and storing it in a new DataFrame.

        Parameters:
        - df (DataFrame): The input DataFrame containing 'Category' and 'Quote' columns.

        Returns:
        - processed_df (DataFrame): A DataFrame with 'category' and 'quote' columns, where 'quote' has been processed.
        """

        processed_quotes = []

        # Process each quote
        for _, row in df.iterrows():
            category = row['Category']
            quote = row['Quote']
            
            processed_text = self.process_text(quote)

            processed_quotes.append({'category': category, 'quote': processed_text})

        # Convert processed_quotes to a DataFrame
        processed_df = pd.DataFrame(processed_quotes)

        return processed_df
    
    def process_text(self, text):
        """
        Processes input text by lowercasing it, and removing non-alphanumeric characters and stop words.

        Parameters:
        - text (str): The input text to be processed.

        Returns:
        - clean_text (str): The processed text.
        """

        # Lowercase input text
        text = text.lower()

        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Split every word in the text into a separate token
        words = word_tokenize(text)

        # Remove stopwords
        clean_words = []
        for word in words:
            if word not in self.stop_words:
                clean_words.append(word)

        return ' '.join(clean_words)