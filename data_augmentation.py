import sys
sys.path.append("..") 

import nltk
from nltk.corpus import wordnet
from data_preprocessing import DataProcessor

nltk.download('wordnet')

def synonym_replacement(words):
    new_words = words.copy()
    for i in range(len(new_words)):
        synonym = get_synonym(new_words[i])
        if synonym:
            new_words[i] = synonym
    return new_words

def get_synonym(word):
    synsets = wordnet.synsets(word)
    if synsets:
        synonyms = synsets[0].lemmas()
        if synonyms:
            return synonyms[0].name()
    return None

data_processor = DataProcessor()
df = data_processor.format_csv('data/data.csv', []).head(10)

for quote in df['Quote']:
    words = quote.split()

    augmented_text = synonym_replacement(words)
    print("Original Text:", quote)
    print("Augmented Text:", ' '.join(augmented_text))
