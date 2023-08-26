from data_preprocessing import DataProcessor
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import PIL.Image

class WordCloudVisualizer:
    def __init__(self, text):
        self.features = text.split()
        data_processor = DataProcessor()
        df = data_processor.format_csv('data/inspiration.csv', ['ID', 'Image-link', 'Quote-url'])
        df = data_processor.process_data(df)
        self.frequencies = {word: 1 for word in self.features}

        for quote in df['quote']:
            quote_list = quote.split()
            print(quote_list)
            for word in self.features:
                self.frequencies[word] += quote_list.count(word)
        print(self.frequencies)

    def get_chart(self):

        plt.figure(figsize=(8, 3))
        if self.features:
            word_cloud_mask = PIL.Image.open('../static/word_cloud_mask.png')
            word_cloud = WordCloud(width=800, height=400, 
                                   background_color='white', contour_color='black',
                                   contour_width=3,
                                   mask=word_cloud_mask).generate_from_frequencies(self.frequencies)

            plt.imshow(word_cloud, interpolation='bilinear')
            plt.axis('off')

        return plt
